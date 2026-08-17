---
type: source
source_path: raw/papers/High-Performance Large-Scale Image Recognition Without Normalization.md
source_kind: paper
title: "High-Performance Large-Scale Image Recognition Without Normalization"
authors: [Andrew Brock, Soham De, Samuel L. Smith, Karen Simonyan]
year: 2021
venue: "arXiv:2102.06171 → ICML 2021"
ingested: 2026-06-17
tags: [nfnet, normalizer-free, agc, batch-normalization, convolutional-neural-network, deepmind]
translation: "[[translations/nfnet]]"
related: ["[[entities/nfnet]]", "[[concepts/convolutional-neural-network]]", "[[entities/convnext]]", "[[entities/convnext-v2]]", "[[questions/cnn-beyond-convnext-v2]]"]
---

# NFNet: 正規化なしの高性能・大規模画像認識

> 原典: [[translations/nfnet]] ・ `raw/papers/High-Performance Large-Scale Image Recognition Without Normalization.md`
> 著者: Andrew Brock, Soham De, Samuel L. Smith, **Karen Simonyan**（DeepMind）
> 出典: arXiv:2102.06171（2021 年 2 月）→ ICML 2021
> コード: <https://github.com/deepmind/deepmind-research/tree/master/nfnets>

> 翻訳メモ: 対応する [[translations/nfnet]] は CLAUDE.md §4 の標準ルールと異なり **Appendix A〜E も翻訳済み**（ユーザーからの個別指示による）。原典 markdown には画像が含まれていなかったため、図 1-8 は arXiv の e-print ソースから生成した。

---

## 一言まとめ

**「BatchNorm を完全に取り除いても、むしろ速く・強くなる」**ことを初めて実証した論文。BN が担っていた 4 つの役割を個別の仕掛けで代替し、大バッチ訓練の不安定さを **AGC（適応的勾配クリッピング）** で解決した。到達点の **NFNet** は追加データなしの ImageNet で **86.5%**（当時 SOTA）、JFT-300M 事前学習で **89.2%** を達成し、**EfficientNet-B7 と同精度を 8.7× 速い訓練で**得る。

**本 wiki にとっての意味**: [[questions/cnn-beyond-convnext-v2]] で「ConvNeXt V2（88.9%）を超える純粋 CNN」として挙げた **NFNet-F4+ 89.2%** の原典。また [[sources/convnext]] が「訓練レシピこそ最大の交絡変数」と示した前年に、**BN そのものを疑った**論文でもある。

---

## 背景と問題意識

### BatchNorm は成功の立役者だが、3 つの実践的欠点を持つ

[[concepts/convolutional-neural-network]] で整理した通り、BN は ResNet 以降の CNN の事実上の標準部品でした。しかし著者らはこう指摘します。

1. **計算が高価**: メモリのオーバーヘッドを生み、一部のネットワークでは勾配評価の時間を大きく増やす
2. **訓練時と推論時で振る舞いが違う**: 隠れたハイパーパラメータが生まれる
3. **ミニバッチ内の訓練事例の独立性を壊す** ← **最も重要**

3 番目の帰結が広範です。ハードウェアをまたぐ正確な再現が難しい、分散訓練で微妙な実装バグの温床になる、**対比学習で情報漏洩を起こす**（SimCLR や MoCo が特別な対策を強いられた理由）、バッチサイズが小さいと性能が落ちるためモデルサイズの上限を縛る——など。

> **補足: なぜ対比学習で問題になるのか** — BN はバッチ内の他の事例の統計量を使うため、モデルが「同じバッチにいる」という情報を手がかりにできてしまう。[[entities/moco]] はデバイス間で事例をシャッフルし、SimCLR は cross-replica BN を使うことでこれを回避している。本 wiki の [[concepts/self-supervised-learning]] で扱う SSL 手法群が BN の扱いに苦労してきた背景がここにあります。

著者らの立場は明確です。「BN は近年の進歩を可能にしたが、**長期的にはむしろ進歩を妨げる**」。

### BN の 4 つの恩恵を分解する（§2）

代替するには、まず何を代替すべきかを知る必要があります。本論文の分析は後続研究にも繰り返し引用される整理です。

| BN の恩恵 | 中身 |
|---|---|
| **残差分岐のダウンスケール** | 初期化時に残差分岐の活性化スケールを縮め、信号をスキップ経路へ偏らせる。これが深いネットワークの訓練を可能にする |
| **平均シフトの除去** | ReLU/GELU は反対称でないため活性化の平均が非ゼロになり、深さに比例して事例間の活性化が相関する。BN はチャネルごとの平均をゼロにしてこれを消す |
| **正則化効果** | バッチ統計量のノイズがテスト精度を高める |
| **効率的な大バッチ訓練** | 損失ランドスケープを滑らかにし、最大の安定学習率を上げる |

---

## 提案手法 1: Normalizer-Free の骨格（先行研究の継承）

本論文は先行研究の **NF-ResNet** を土台にします。残差ブロックを次の形にする。

$$h_{i+1}=h_{i}+\alpha f_{i}(h_{i}/\beta_{i})$$

- $f_i$ は初期化時に**分散保存的**（$\text{Var}(f_i(z))=\text{Var}(z)$）
- $\alpha$（既定 0.2）が分散の増加率を決める
- $\beta_i=\sqrt{\text{Var}(h_i)}$ は**解析的に予測**できる（$\text{Var}(h_{i+1})=\text{Var}(h_i)+\alpha^2$）
- 遷移ブロックでは分散を $1+\alpha^2$ に**リセット**（BN 版の挙動を模倣）
- **SkipInit**: ゼロ初期化の学習可能スカラーを各残差分岐末尾に置き、ブロックを恒等写像から始める

**平均シフト**は **Scaled Weight Standardization** で消します。

$$\hat{W_{ij}}=\frac{W_{ij}-\mu_{i}}{\sqrt{N}\sigma_{i}}$$

活性化関数も非線形性ごとのゲイン $\gamma$ でスケールし、全体が分散保存的になるようにします（ReLU なら $\gamma=\sqrt{2/(1-1/\pi)}$）。

**ただしここまでで足りない**: NF-ResNet は BN 版に匹敵はするものの、**バッチサイズ 4096 以上で不安定**になり、EfficientNet の性能にも届きませんでした。これが本論文の出発点です。

---

## 提案手法 2: AGC（Adaptive Gradient Clipping）— 本論文の中核

### 発想

素朴な勾配クリッピング（$\|G\|>\lambda$ ならスケール）は、**閾値 $\lambda$ に極端に敏感**で、深さ・バッチサイズ・学習率を変えるたびに再調整が必要でした。

著者らの洞察は、**勾配ノルムそのものではなく「1 ステップで重みがどれだけ変わるか」を見るべき**というものです。モメンタムなしの勾配降下なら:

$$\frac{\|\Delta W^{\ell}\|}{\|W^{\ell}\|}=h\frac{\|G^{\ell}\|_{F}}{\|W^{\ell}\|_{F}}$$

つまり **勾配ノルム / 重みノルムの比**が、更新の相対的な大きさを表します。これが大きいと訓練が不安定になる。そこでこの比でクリップします。

$$
G^{\ell}_{i}\rightarrow\begin{cases}\lambda\frac{\|W^{\ell}_{i}\|^{\star}_{F}}{\|G^{\ell}_{i}\|_{F}}G^{\ell}_{i}&\text{if}\ \frac{\|G^{\ell}_{i}\|_{F}}{\|W^{\ell}_{i}\|^{\star}_{F}}>\lambda\\
G^{\ell}_{i}&\text{otherwise}\end{cases}
$$

**層単位ではなくユニット単位**（重み行列の行ごと）に適用するのが経験的に良い、というのが実装上の要点です。$\|W_i\|^\star_F=\max(\|W_i\|_F,\epsilon)$（$\epsilon=10^{-3}$）でゼロ初期化パラメータの勾配が常にゼロになるのを防ぎます。

> **補足: LARS との関係** — AGC は「正規化されたオプティマイザ」（LARS 等）の**緩和版**と位置づけられています。LARS が勾配の大きさを完全に無視して更新ノルムをパラメータノルムの一定比にするのに対し、AGC は**上限だけを課し、下限は課さない**。著者らは LARS でも大バッチ訓練は安定するが性能が落ちると報告しています。

### アブレーション（§4.1）

<figure>

![](../../raw/assets/nfnet/fig2.png)

<figcaption>図2: (a) AGC は NF-ResNets をより大きなバッチサイズへ効率的にスケールする。(b) 異なるクリッピング閾値 λ にわたる性能。</figcaption>
</figure>

- **AGC により NF-ResNet がバッチ 4096 まで BN 版と同等以上を維持**できるようになる
- **バッチが大きいほど小さい（強い）$\lambda$ が必要**
- **バッチサイズが小さいときは AGC の恩恵は小さい**（そもそも不安定でないため）
- **最終線形層はクリップしない方が常に良い**。逆に 4 つのステージすべての重みはクリップしないとバッチ 4096 で安定しない

---

## 提案手法 3: NFNet アーキテクチャ

<figure>

![](../../raw/assets/nfnet/fig3.png)

<figcaption>図3: NFNet の bottleneck ブロック設計。入力に 1/β を掛けて 1×1 → 3×3 → 3×3 → 1×1 を通し、出力に α を掛けてスキップ経路と加算する。Stage Widths は ResNet の [256, 512, 1024, 2048] に対し NFNet は [256, 512, 1536, 1536]。Stage Depths は ResNet の [3, 4, 6, 3] に対し NFNet は [1, 2, 6, 3] × N。</figcaption>
</figure>

出発点は **GELU 付き SE-ResNeXt-D**。ここに 4 つの変更を加えます。

| 変更 | 内容と理由 |
|---|---|
| **グループ幅を 128 に固定** | ブロック幅によらず固定。小さいグループ幅は FLOPS を減らすが**計算密度が落ちて実速度が上がらない**（TPUv3 では グループ幅 8 と 128 が同速） |
| **深さパターン $[1,2,6,3]\times N$** | ResNet の非一様な深さ増加（第 2・3 ステージだけ増やす）は最適でない。**単純に全ステージを $N$ 倍**する |
| **幅パターン $[256,512,1536,1536]$** | 多数の候補のうち既定より良いのはこれだけ。**第 3 ステージに容量を足す**のが最良 |
| **$3\times3$ グループ化 conv をもう 1 つ追加** | FLOPS への影響は最小で訓練時間もほぼ変わらないのに精度が上がる |

**設計の指針が独特**です。著者らは理論的 FLOPS でも推論レイテンシでもなく、**実機（TPUv3 / V100）の訓練レイテンシ**を最適化対象に選びました。「EffNet-B0 は ResNet-50 の 1/10 の FLOPS なのに訓練速度も最終性能も同程度」という観察が動機です。

**スケーリングは深さと解像度のみ**（幅はスケールしない）。EfficientNet の複合スケーリングは細い MobileNet 系には効くが、ResNet 系バックボーンでは幅スケーリングが効かないため。推論は訓練解像度の約 1.33 倍で行い、**その解像度でのファインチューニングはしない**。

**表**: NFNet ファミリー（[[translations/nfnet]] 表1）

| Variant | Depth | Dropout | Train | Test |
|---|---|---|---|---|
| F0 | [1, 2, 6, 3] | 0.2 | 192px | 256px |
| F1 | [2, 4, 12, 6] | 0.3 | 224px | 320px |
| F2 | [3, 6, 18, 9] | 0.4 | 256px | 352px |
| F3 | [4, 8, 24, 12] | 0.4 | 320px | 416px |
| F4 | [5, 10, 30, 15] | 0.5 | 384px | 512px |
| F5 | [6, 12, 36, 18] | 0.5 | 416px | 544px |
| F6 | [7, 14, 42, 21] | 0.5 | 448px | 576px |

---

## 実験結果と知見

### ImageNet（追加データなし）

<figure>

![](../../raw/assets/nfnet/fig1.png)

<figcaption>図1: ImageNet 検証精度 vs 訓練レイテンシ（TPUv3）。赤線の NFNet が EffNet / BoTNet / LambdaNet / DeiT のいずれのパレートフロントも上回る。NFNet-F1 は EffNet-B7 と同精度を 8.7× 速い訓練で達成し、NFNet-F5 は EffNet-B7 と同程度のレイテンシで 86.0% に達する。</figcaption>
</figure>

| モデル | params | Top-1 | TPUv3 訓練 |
|---|---|---|---|
| EffNet-B7 | 66.0M | 84.7 | 1397.0 ms |
| **NFNet-F1** | 132.6M | **84.7** | **158.5 ms（8.7× 高速）** |
| EffNet-B8 + MaxUp | 87.4M | 85.8 | — |
| **NFNet-F5** | 377.2M | **86.0** | 1398.5 ms |
| **NFNet-F5 + SAM** | 377.2M | **86.3** | 1958.0 ms |
| **NFNet-F6 + SAM** | 438.4M | **86.5** | 2774.1 ms |

**NFNet-F1 が EffNet-B7 と同精度を 8.7× 速く**という結果が本論文の看板です。FLOPS では NFNet の方がはるかに多い（35.5B vs 37.0B は同程度だが F5 は 289.8B）にもかかわらず実機では速い——**FLOPS と実速度の乖離**を突いた設計思想が効いています。

### データ拡張の寄与が大きい（表2）

| 段階 | F0 | F3 |
|---|---|---|
| Baseline | 80.4 | 82.3 |
| + Modified Width | 80.9 | 82.3 |
| + Second Conv | 81.3 | 82.7 |
| + MixUp | 82.2 | 83.5 |
| + RandAugment | **83.2** | **85.0** |
| + CutMix | **83.6** | **85.7** |

**アーキテクチャ変更の寄与（+0.9 / +0.4）よりデータ拡張の寄与（+2.3 / +3.0）の方がはるかに大きい**のが率直に読み取れます。RA は 4 層（一般的な既定の 2 層より強い）で、**画像解像度に応じて強度をスケールするのが重要**（強すぎると画像が真っ白になる）。

**重要な観察**: この強い拡張の組み合わせは **NFNet には効くが EfficientNet には効かない**。著者らは「NFNet は BN の暗黙的正則化を欠くため、強い拡張にも大規模事前学習にも適している」と説明します。

### BN 版との直接比較（Appendix C 表6）

同じアーキテクチャを BN で訓練すると:

- **精度はわずかに低い**（F0 で 83.6 vs 83.4）
- **訓練は 20-40% 遅い**（F0 で 73.3ms vs 111.7ms）
- **F4 / F5 は AGC の有無にかかわらず訓練が不安定**（bfloat16 と BN 統計量の相互作用が原因と推測）

### 転移学習 — NF の真価はここ

**JFT-300M（3 億枚、非公開）で事前学習 → ImageNet でファインチューニング**:

| モデル | 224px | 320px | 384px |
|---|---|---|---|
| BN-ResNet-50 | 78.1 | 79.6 | 79.9 |
| **NF-ResNet-50** | **79.5** | **80.9** | **81.1** |
| BN-ResNet-200 | 81.8 | 83.1 | 83.5 |
| **NF-ResNet-200** | **82.9** | **84.1** | **84.3** |

**全ケースで NF が BN を約 1 ポイント上回る**。著者らの仮説は「大規模事前学習では BN の暗黙的正則化がむしろ有害で、モデルが訓練セットに容量を割く能力を削いでいる」。

そして **NFNet-F4+ が 89.2%**（[[questions/cnn-beyond-convnext-v2]] で引用した数字）。当時、追加データを使った手法として 2 位（1 位は EffNet-L2 + Meta Pseudo Labels の 90.2%）、**転移学習で達成された最高精度**でした。

> **計算効率も注目に値する**: NFNet-F4+ は **1.86k TPUv3-core-days** で 89.2% に到達。EffNet-L2 + Meta Pseudo Labels は 90.2% に **22.5k**（12 倍）、EffNet-L2 + NoisyStudent + SAM は 88.6% に 12.3k を要しています。

---

## Appendix 由来の非自明な知見

- **大規模事前学習では weight decay をゼロから始めよ**（A.5）: 「大きな weight decay から始めて減らすのではなく、**ゼロから始めて軽く増やす**ことを推奨する」。この設定では Adam より SGD が良いとも報告。ViT 論文と正反対の結論で、著者らは自分たちの BN-ResNet ベースラインが先行研究より大幅に強い（77.54 → 79.9）ことをその証拠として挙げています
- **大きいモデルを短く訓練する方が効率的**（A.5）: 幅広の F4+ を 20 エポックと、F4 を 40 エポックが同性能。訓練レイテンシは同じ（約 830ms/step）
- **SAM を 20-40% 増で済ませる工夫**（A.4）: 上昇ステップの勾配計算に**バッチの 20% だけ**使えば性能は同等。ただし上昇と降下で**別のバッチを使うと効果が完全に消える**
- **weight decay を適用しない場所**（A.1）: weight-standardized 層のアフィンゲイン・バイアス、SkipInit ゲインには適用しない
- **否定的結果が丁寧**（Appendix E）: ステージごとにグループ幅や bottleneck 比を変える異質な設計は「**不要であり、解釈可能な設計パターンの推論を混乱させる**」。DCT 係数上での積極的ダウンサンプリングは、情報を失わない可逆変換でも精度を犠牲にする。$1\times1$ conv のグループ化は S&E で補っても実レイテンシを削減できない

---

## 限界・批判的視点

- **パラメータ効率が悪い**。NFNet-F1 は 132.6M で 84.7%、EffNet-B7 は 66.0M で同じ 84.7%。**パラメータは 2 倍**。著者自身「FLOPs 対精度では競争力があるが、**パラメータ対精度ではそうではない**」と明記しています
- **最高値は非公開データ依存**。89.2% は JFT-300M（Google 内部の 3 億枚）を使ったもの。[[sources/convnext-v2|ConvNeXt V2]] の 88.9% が「**公開データのみ**」であることと直接比較はできません（[[questions/cnn-beyond-convnext-v2]] 参照）
- **attention を一切試していない**。著者らは Appendix E の最後で「attention の変種を採用すれば結果はさらに改善される可能性が高い」と自ら述べています。実際その後 CV は attention 側へ大きく動きました
- **BN 排除が主流にはならなかった**。翌年の [[sources/convnext|ConvNeXt]] は BN を捨てましたが、**採用したのは LayerNorm** であって Normalizer-Free ではありません。NFNet の Scaled Weight Standardization + AGC + SkipInit という組み合わせは、**部品が多く調整項目も多い**（$\alpha$、$\beta$、$\lambda$、$\epsilon$、γ ゲイン）ため、LN に置き換える方が単純だったと言えます
- **AGC は完全にハイパラフリーではない**。$\lambda$ はバッチサイズ・学習率・オプティマイザに依存し、著者も「バッチが大きいほど小さくすべき」と述べています。素朴なクリッピングより頑健になっただけで、調整は残ります
- **訓練レイテンシ最適化という前提の賞味期限**。「現在のアクセラレータでは EfficientNet の低 FLOPS が活きない」という前提は 2021 年の TPUv3/V100 のもの。ハードウェアが変われば結論も動きます（[[sources/convnext]] が A100 で ConvNeXt が Swin 比 +49% と示したのと同じ構図）

---

## 用語と略称

- **NFNet** = Normalizer-Free Network。本論文が提案したモデルファミリー（F0〜F6）
- **NF-ResNet** = Normalizer-Free ResNet。本論文が土台にした先行研究のアーキテクチャ
- **AGC** = Adaptive Gradient Clipping（適応的勾配クリッピング）。勾配ノルム / 重みノルムの**ユニット単位**の比でクリップする
- **BN** = Batch Normalization。本論文が排除の対象とするもの
- **Scaled Weight Standardization** = 畳み込みの重みを平均 0・分散正規化して再パラメータ化し、平均シフトを消す技法
- **平均シフト（mean shift）** = ReLU/GELU が反対称でないために活性化の平均が非ゼロになり、深さに比例して事例間の活性化が相関する現象
- **SkipInit** = 各残差分岐末尾のゼロ初期化された学習可能スカラー。ブロックを恒等写像から始める
- **SE-ResNeXt-D** = Squeeze-Excitation + ResNeXt + D 型ダウンサンプリング。NFNet の出発点
- **SAM** = Sharpness-Aware Minimization。平坦な最小値を探す最適化手法。既定では訓練時間が倍になる
- **LARS** = Layer-wise Adaptive Rate Scaling。AGC が「緩和版」として位置づける正規化オプティマイザ
- **RA / RandAugment** = 自動データ拡張。本論文は 4 層で使用
- **MixUp / CutMix** = 画像を混合／貼り付けるデータ拡張
- **JFT-300M** = Google 内部の 3 億枚・約 18,000 クラスの非公開データセット
- **TPUv3-core-days** = TPUv3 コア × 日数で測る計算コストの単位

## 関連ページ

- [[entities/nfnet]] — NFNet のスペック・F0〜F6 構成・系譜
- [[questions/cnn-beyond-convnext-v2]] — 本論文の 89.2% を「ConvNeXt V2 を超える純粋 CNN」として引用した query
- [[concepts/convolutional-neural-network]] — CNN の系譜と部品。**BN vs LN** の議論の背景
- [[sources/convnext]] / [[entities/convnext]] — 翌年に BN → **LN** で正規化問題に別解を出した論文。「訓練レシピが最大の交絡変数」という発見も対で読む価値がある
- [[sources/convnext-v2]] — BN がマスク入力と相性が悪く 80.5 まで落ちる、という別角度からの BN 批判
- [[sources/maxvit]] — NFNet-F4+ 89.20 を比較表に載せている（本 wiki が最初に NFNet を知った経路）
- [[concepts/self-supervised-learning]] — BN が対比学習で情報漏洩を起こす問題の文脈
- [[entities/moco]] — BN の情報漏洩をデバイス間シャッフルで回避した例
