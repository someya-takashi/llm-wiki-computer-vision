---
type: source
source_path: raw/papers/DenseNets Reloaded_Paradigm Shift Beyond ResNets and ViTs.md
source_kind: paper
title: "DenseNets Reloaded: Paradigm Shift Beyond ResNets and ViTs"
authors: [Donghyun Kim, Byeongho Heo, Dongyoon Han]
year: 2024
venue: ECCV 2024
ingested: 2026-08-31
tags: [rdnet, densenet, concatenation, skip-connection, convolutional-neural-network, backbone, naver, modernization]
translation: "[[translations/rdnet]]"
---

# RDNet: DenseNet を近代化して ResNet 系を超える

> 原典: [[translations/rdnet]] ・ `raw/papers/DenseNets Reloaded_Paradigm Shift Beyond ResNets and ViTs.md`
> 著者: Donghyun Kim*, Byeongho Heo, Dongyoon Han（NAVER Cloud AI / NAVER AI Lab）
> 出典: arXiv:2403.19588 → **ECCV 2024**
> コード: <https://github.com/naver-ai/rdnet>

## 一言まとめ

**[[sources/convnext|ConvNeXt]] が ResNet に対してやったことを、DenseNet に対してやった論文。** ただし主張はもう一段深く、**「加算ショートカットより連結ショートカットの方が本来強い」**という命題を **15,000 個のランダムネットワーク**で検証している。

## 背景と問題意識

### wiki に欠けていた軸 — ショートカットの「型」

本 wiki の CNN の系譜（[[concepts/convolutional-neural-network]]）は、ResNet → ResNeXt → DenseNet → MobileNet → EfficientNet → ConvNeXt と並んでいるが、**DenseNet だけが「全層間の密な接続」という 1 行で片づけられていた**。本論文はそこに正面から光を当てる。

| | ResNet 型 | DenseNet 型 |
|---|---|---|
| **定式化** | $\mathbf{X}_{l+1}=\mathbf{X}_{l}+f(\mathbf{X}_{l}\mathbf{W})$ | $\mathbf{X}_{l+1}=[\mathbf{X}_{l},\ f(\mathbf{X}_{l}\mathbf{W})]$ |
| **ショートカット** | **加算**（additive） | **連結**（concatenation） |
| **利点** | 恒等写像の微分が常に 1 → 勾配消失を回避、モジュール化しやすい | **特徴の再利用**、パラメータ効率、初期層への明示的な教師の伝播 |
| **その後** | ResNeXt → EfficientNet → **ConvNeXt** → ViT → **Swin** と系譜が続く | **人気が衰退した** |

論文の指摘が鋭い。**ViT も Swin も ConvNeXt も、内部のショートカットはすべて加算である。** つまり「Transformer 対 CNN」という対立軸の下では、**残差学習という同じ土台を全員が共有していた**。本論文はそこを問い直す。

### なぜ DenseNet は消えたのか

論文の診断は 2 点。

1. **幅のスケーリングがメモリを食う。** 連結は層ごとに特徴が積み上がるので、DenseNet-161/-233 のような広いモデルはメモリを大量に消費する
2. **訓練レシピと設計要素が古いまま放置された。** 「手つかずの訓練手法と伝統的な設計要素がその能力を十分に引き出していなかった」

**2 点目は [[sources/convnext]] とまったく同じ診断である。** ConvNeXt は「ViT が ResNet より強いとされた差の多くは、比較対象の ResNet が 2015 年のレシピの古い数字だったことに由来する」と示した。本論文はそれを DenseNet に対して繰り返す。

## 提案手法

### 予想 — 連結はランクを増やす

理論的な動機づけが 2 段構えになっている。

**(1) 連結はランクを増やす効果的な方法である。** 層の出力 $f(\mathbf{X}\mathbf{W})$（$\mathbf{W}\in\mathbb{R}^{d_{in}\times d_{out}}$）について、$d_{in}$ がそれほど小さくなければ非線形性のために $\mathrm{rank}(f(\mathbf{X}\mathbf{W}))$ は $d_{out}$ に近づく。**$d_{out}>d_{in}$ である層は増加した表現能力を提供する**——連結はまさに出力次元を入力より大きくする操作である。

**(2) 戦略的な設計はメモリの懸念を緩和する。** 積み重ねた層 $\mathbf{X}\mathbf{W}_{1}f(\mathbf{W}_{2})$（$d_r < d_{in}, d_{out}$）でも、$d_r$ がそれほど小さくなければランクは保たれる。**つまり中間の次元削減器（遷移層）を頻繁に挟んでもランクは大きく損なわれない。** これが後の「遷移層を増やす」という設計に直結する。

> **メモリ問題とランク保持のトレードオフを、「遷移層を増やす」という 1 つの操作で同時に解いている**のが本論文の設計上の要点である。

### パイロット研究 — 15,000 個のランダムネットワーク

**予想の検証方法が本論文で最も特徴的な部分である。**

**Tiny-ImageNet 上で 15,000 個を超えるランダムネットワークをサンプルし、ショートカットだけを加算/連結で入れ替えて比較する。** パラメータ数・FLOPs・メモリを揃え、深さ・幅・活性化（ReLU/SiLU/Mish/GELU/LeakyReLU）・正規化（BN/LN）・カーネルサイズ（3/5/7/9）を振り、さらにブロック型（PreNorm / PostNorm / PostNorm 活性化なし）とデータ拡張・オプティマイザまで変える。

**表**: パイロット研究の結果（原典 表7）

| Model | Skip | Top-1 (%) |
|---|---|---|
| RandNet$_\mathcal{A}$（2M） | add | 45.8±2.0 |
| RandNet$_\mathcal{A}$ | **concat** | **47.5±2.1** |
| RandNet$_\mathcal{B}$（4M） | add | 50.1±2.1 |
| RandNet$_\mathcal{B}$ | **concat** | **51.2±2.1** |
| RandNet$_\mathcal{C}$（9M） | add | 53.2±2.2 |
| RandNet$_\mathcal{C}$ | **concat** | **54.3±2.0** |
| RandNet$_\mathcal{D}$（+ 拡張） | add | 57.4±1.5 |
| RandNet$_\mathcal{D}$ | **concat** | **58.1±1.4** |
| RandNet$_\mathcal{E}$（+ AdamW） | add | 58.2±1.5 |
| RandNet$_\mathcal{E}$ | **concat** | **58.9±1.6** |

<figure>

![](../../raw/assets/rdnet/fig3.png)

<figcaption>図4（再掲）: 訓練されたモデルの累積確率 vs 誤差。赤が Concat、緑が Add。5 つの設定すべてで赤の曲線が左に位置し、より低い誤差に集中している。</figcaption>
</figure>

**全 10 設定で連結が勝つ。** 平均差は約 1 ポイントで標準偏差（約 2）より小さいが、**分布全体が系統的にシフトしている**のが図4 から読める。

> **この検証設計は RegNet（Radosavovic ら）の「設計空間を population として評価する」手法の応用である。** 単一のモデルを 2 つ作って比べるのではなく、**設計空間全体から無作為抽出した分布同士を比べる**ことで、「たまたまチューニングが上手くいっただけ」という交絡を潰している。[[sources/l2rw]] が「乱数ベースラインを置け」と警告した問題への、別方向からの回答とも読める。

**ただし正直な但し書きもある**（Appendix 0.I.1）。

> **データ拡張が加算的ショートカットと連結的ショートカットの間の性能差を縮める。特に AdamW で stochastic depth を用いて訓練された要素は加算的ショートカットからより恩恵を受ける。**

実際 Appendix 表K では、$\mathcal{E}$ 空間の Sto. Depth（57.5 vs 57.5）と RandErase（58.1 vs 58.1）で**差が消えている**。**近代的な訓練レシピは連結の優位を部分的に打ち消す**——これは本論文自身の主張（「訓練レシピが古かったから DenseNet が沈んだ」）と表裏一体の発見である。

### 近代化ロードマップ

**[[sources/convnext]] と同じ形式の段階表**が本論文の心臓部である。

**表**: ImageNet-1K の性能の進行（原典 表1）

| | 要素 | Top-1 | Param (M) | FLOPs (G) | Lat b1 (ms) | Mem (GB) |
|---|---|---|---|---|---|---|
| (a) | **DenseNet-201**（現代的レシピで再訓練） | 79.7 | 20.0 | 4.3 | **38.4** | 3.9 |
| (b) | + **より広く浅く**（GR 32 → 120） | 79.5 (−0.2) | 21.8 | 11.1 | **8.5 (−29.9)** | 3.2 |
| (c) | + **近代化されたブロック**（LN / depthwise / 7×7） | 80.4 (+0.9) | 12.9 | 4.8 | 10.4 | 3.4 |
| (d) | + **中間チャネル次元↑（GR↓）** | 80.8 (+0.4) | 19.9 | 4.7 | 11.8 | 3.1 |
| (e) | + **遷移層↑（GR↑）** | **82.3 (+1.5)** | 21.2 | 5.0 | 11.0 | 3.4 |
| (f) | + **パッチ化ステム** | 82.4 (+0.1) | 21.2 | 4.9 | 11.0 | 3.2 |
| (g) | + **洗練された遷移層** | 82.6 (+0.2) | 22.4 | 4.9 | 13.6 | 3.1 |
| (h) | + **チャネル再スケーリング** | **82.8 (+0.2)** | 23.9 | 5.0 | 14.0 | 3.1 |

**読みどころが 3 つある。**

**(1) 最初のステップ (b) は精度を下げるが、レイテンシを 38.4 → 8.5ms と 4.5 倍速くする。** 「より広く浅く」は精度のためではなく**速度のため**の変更であり、そのうえで後続のステップが精度を積み上げる。**ConvNeXt のロードマップでも depthwise 化が一旦 80.5 → 78.3 と下げてから幅で取り戻す**という同じ形が出ていた。

**(2) 最大の跳ね（+1.5）は「遷移層を増やす」である。** 各ステージの後だけでなく **3 ブロックごとにストライド 1 の遷移層を挟む**。連結で膨れる特徴を頻繁に圧縮することで、**高い GR を維持できるようになる**——予想 (2) がそのまま設計になっている。ステージごとに GR を変える（64, 104, 128, 224）のもここで導入される。

**(3) ER を GR から切り離したのが (d)。** DenseNet は拡張率を GR に掛けていた（ER = 4 × GR）が、**RDNet は入力次元に掛ける**。「これが非線形性を通した符号化された特徴の能力を損なう」という指摘で、予想 (1) のランクの議論と整合する。

### アーキテクチャ

<figure>

![](../../raw/assets/rdnet/fig1.png)

<figcaption>図1（再掲）: RDNet の模式図。4 ステージ構成で、各 Stage-N は L_N 個の混合ブロックからなる。混合ブロックは 3 つの特徴混合器 f と 1 つの遷移層で構成される（最後の混合ブロックは遷移層を持たない）。特徴混合器は DWConv 7×7 → LayerNorm → Linear（C→4C）→ GELU → Linear（4C→GR）で、出力は GR 次元。遷移層は LayerNorm → Conv s×s stride s でチャネルを半分にしつつダウンサンプリングする。丸 C が連結を表す。</figcaption>
</figure>

**特徴混合器の出力が GR 次元である**のが本設計の肝である。ConvNeXt のブロックが入力と同じ次元を返して加算するのに対し、**RDNet のブロックは GR 次元だけ返して連結される**。だから GR が「1 ブロックあたり何チャネル増えるか」を決める。

| モデル | GR | B（特徴混合器の数） |
|---|---|---|
| **RDNet-T** | (64, 104, 128, 224) | (3, 3, 12, 3) |
| **RDNet-S** | (64, 128, 128, 240) | (3, 3, 21, 6) |
| **RDNet-B** | (96, 128, 168, 336) | (3, 3, 21, 6) |
| **RDNet-L** | (128, 192, 256, 360) | (3, 3, 24, 6) |

## 実験結果と知見

### ImageNet — マイルストーンを上回る

**表**: マイルストーンとの比較（原典 表4 より抜粋。Lat は 128 枚の推論時間 ms）

| Model | Param | FLOPs | Lat | Top-1 |
|---|---|---|---|---|
| Swin-T | 28 | 4.5 | 173 | 81.3 |
| ConvNeXt-T | 29 | 4.5 | 150 | 82.1 |
| DeiT III-S | 22 | 4.6 | 128 | 81.4 |
| **RDNet-T** | **24** | 5.0 | 175 | **82.8** |
| Swin-B | 89 | 15.4 | 445 | 83.5 |
| ConvNeXt-B | 89 | 15.4 | 417 | 83.8 |
| DeiT III-B | 87 | 17.5 | 422 | 83.8 |
| **RDNet-B** | 87 | 15.4 | 472 | **84.4** |
| ConvNeXt-L (384²) | 198 | 101.1 | 2550 | 85.5 |
| DeiT III-L (384²) | 304 | 191.2 | 4586 | 85.8 |
| **RDNet-L (384²)** | **186** | 101.9 | **2714** | **85.8** |

**要旨の「Swin Transformer、ConvNeXt、DeiT-III を上回る」は正確である。** 特に **RDNet-L(384²) が DeiT-III-L(384²) と同じ 85.8 を、パラメータ 186M 対 304M・レイテンシ 2714 対 4586ms で達成する**のは印象的。

### 最新モデルとの比較では「精度で負けて速度で勝つ」

**表**: 最新モデルとの比較（原典 表2 より抜粋）

| Model | Top-1 | **b1 (ms)** | b128 (ms) | Mem |
|---|---|---|---|---|
| **RDNet-T** | 82.8 | **7.4** | 175.2 | 4.1 |
| HorNet-T | 82.8 | 21.2 | 183.7 | 4.1 |
| BiFormer-S | **83.8** | 51.6 | 197.2 | 8.5 |
| SMT-S | 83.7 | 46.9 | 335.3 | 5.3 |
| **RDNet-S** | 83.7 | **11.9** | 289.0 | 5.4 |
| **RDNet-B** | **84.4** | **11.7** | 471.6 | 6.9 |
| MogaNet-XL | **85.1** | 66.3 | 1512.5 | 24.1 |
| **RDNet-L** | 84.8 | **15.7** | 933.7 | 10.9 |

論文自身が正直に書いている——**「我々のモデルは精度でわずかに劣るものの、優れた速度の指標で大きく埋め合わせる」**。

> **バッチサイズ 1 のレイテンシの差が異常である。** RDNet-T は 7.4ms、BiFormer-S は 51.6ms で **7 倍**。RDNet-B に至っては 11.7ms で、より小さい MogaNet-S（20.0ms）より速い。**これは FLOPs や パラメータ数では見えない差**で、本 wiki の [[sources/convnext]] が「A100 で Swin 比 +49% スループット」を強調したのと同じ論点——**実機のレイテンシは理論指標と一致しない**——のさらに極端な例である。
>
> ただし **b128 では差が縮む**（RDNet-T 175.2 vs HorNet-T 183.7）。**RDNet の速度優位は小バッチ・低レイテンシ領域に偏っている。**

### 下流タスクと転移性

| タスク | RDNet | ConvNeXt | Swin |
|---|---|---|---|
| **ADE20K mIoU (ss)**（-B） | **49.6** | 49.1 | 48.1 |
| **COCO AP^box**（-T, Mask-RCNN 3x） | **47.5** | 46.2 | 46.0 |
| **ゼロショット CLIP**（-B） | **54.1** | 51.2 | – |

**COCO では RDNet-T が 43M パラメータで ConvNeXt-T の 48M を 1.3 ポイント上回る。** DenseNet の系譜が密予測タスクで強いという歴史（[[entities/hrnet]] の文脈で言えば「解像度と特徴の保持」）と整合する。

**Appendix 0.F の転移性が最も驚く結果である。**

| Model | Param | iNat18 | iNat19 |
|---|---|---|---|
| DeiT-III-S | 22 | 67.1 | 72.7 |
| **RDNet-T** | **24** | **77.0** | **81.2** |
| DeiT-III-L | **304** | 75.6 | 79.3 |
| **RDNet-L** | 186 | **81.5** | **83.7** |

**RDNet-T（24M）が iNaturalist で DeiT-III-L（304M）を上回る。** 13 倍のパラメータ差を覆している。長尾分類（クラスあたりのサンプル数が極端に偏る設定）で連結型が強いというのは、**特徴の再利用が少数クラスの学習を助ける**という DenseNet 本来の主張と符合する。

### 頑健性 — 精度で負けても OOD で勝つ

**表**: OOD ベンチマーク（原典 表I より抜粋）

| Model | IN | **Avg Shift** | Sketch | R |
|---|---|---|---|---|
| NAT-T | **83.2** | 44.0 | 31.9 | 44.9 |
| **RDNet-T** | 82.8 | **44.7** | **37.0** | **49.0** |
| HorNet-S | **84.0** | 47.3 | 36.9 | 49.7 |
| **RDNet-S** | 83.7 | 47.8 | **39.8** | **52.8** |
| ConvNeXt-L | 84.3 | 49.9 | 40.1 | 53.5 |
| **RDNet-L** | **84.8** | **52.2** | **44.5** | **56.5** |

> **論文が明示している通り「RDNet が競合モデルより ImageNet-1K で低い精度を示す場合でさえ、高い OOD スコアを達成する」。** ImageNet の精度と分布シフトへの頑健性が乖離する例で、**ImageNet top-1 だけでモデルを選ぶことの危うさ**を示す。RDNet-L は ConvNeXt-L に対し IN で +0.5 なのに Avg Shift で **+2.3**、Sketch で **+4.4** 開く。

### 解像度ロバスト性

<figure>

![](../../raw/assets/rdnet/fig4.png)

<figcaption>図5（再掲）: 解像度に対する精度/レイテンシ/メモリ。画像サイズ 1000 において RDNet-T が約 68% を保つのに対し、DeiT-S は約 34%、DenseNet161 は約 44% まで落ちる。中央・右のパネルでは RDNet が ConvNeXt/Swin と同様の緩やかな増加に留まり、DenseNet161 が急激に悪化するのと対照的。</figcaption>
</figure>

**論文が拾った非自明な観察**: 「**DenseNet161 は強いデータ拡張なしで訓練されているにもかかわらず適応性を享受し、強いデータ拡張で訓練された DeiT-S を上回る。我々はこれを密な接続の有効性に帰する。**」

同時に、**DenseNet161 はレイテンシとメモリで破綻する**（画像サイズ 800 でメモリ 27GB 対 RDNet-T 17GB）。**RDNet は「DenseNet の解像度ロバスト性を保ちつつ、DenseNet のメモリ問題を解いた」**という位置づけになる。

### CKA — 連結型は各層で違うものを学ぶ

<figure>

![](../../raw/assets/rdnet/fig5.png)

<figcaption>図6（再掲）: CKA 解析。左が RDNet-T の層間類似度、中央が ConvNeXt-T、右が両者の間。RDNet は対角以外が全体に暗く各層で異なる特徴を学習している一方、ConvNeXt は中盤の層同士が明るく似た特徴を繰り返し学習している。</figcaption>
</figure>

**ConvNeXt の中盤の層は互いに似た特徴を学んでいる**（明るいブロック）のに対し、**RDNet は各層が異なる特徴を学ぶ**。これは連結の設計と整合する——**加算は同じ空間に足し込むので表現が似通いやすいが、連結は新しいチャネルを足すので分化しやすい**、という読み方ができる。

### Stochastic Depth の再訪

**DenseNet はもともと Stochastic Depth を使っていなかった**（接続パターンが似ているため不要と考えられた）。しかし:

| Ratio | Top-1 |
|---|---|
| 0 | 81.6 |
| **0.15** | **82.8** |

**0 → 0.15 で +1.2%p。** これは表1 のどの単一ステップより大きい。**「使わないのが定石」とされていた要素を再検証したら最大の利得だった**という、[[sources/convnext]] 型の教訓の好例である。

## 限界・批判的視点

**論文自身が認めている点**

- **-large までしかスケールしていない**。「資源の制約により ViT-G のような上位のスケールへのさらなる拡張ができていない」
- 最新モデル（BiFormer, SMT, MogaNet）には**精度で負ける**
- ConvNeXt ほど**網羅的なハイパーパラメータ探索をしていない**（Appendix 0.A.2 で「はるかに軽くスイープする」と明記、0.E でも同様の但し書き）

**本 wiki の視点から見た限界**

- **パイロット研究の差は小さい。** 平均差は約 1 ポイントで標準偏差（約 2）より小さく、**Appendix 0.I.1 では近代的な訓練レシピ（AdamW + stochastic depth / RandErase）で差が消える**ケースが記録されている。「連結が本質的に強い」という主張と、「近代的レシピが差を埋める」という自身の発見は緊張関係にある。
- **メモリでは負けている。** RDNet-T は 4.1GB に対し ConvNeXt-T は 2.7GB、Swin-T は 2.6GB。表1 の進行でも訓練メモリは 3.9 → 3.1GB としか改善せず、**「メモリ効率を高める」という要旨の主張は DenseNet 比では正しいが ResNet 系比では成立していない**。
- **速度優位がバッチサイズに依存する**。b1 で 7 倍速いのに b128 では並ぶ。実務でどちらが効くかは配備条件次第である。
- **CKA の解釈が弱い。** 「各層で異なる特徴を学ぶ」ことがなぜ良いのかは論じられていない。多様性が高いこと自体は必ずしも性能に直結しない。
- **[[entities/hrnet]] との関係が扱われていない。** HRNet も「特徴を捨てずに保持して融合する」設計で、密予測で強い。連結型ショートカットの系譜として並べる価値があるが本論文は言及しない。

## 既存 wiki との接続

**[[sources/convnext]] と対をなす論文である。** 同じ手続き（現代的レシピでベースラインを引き直し、設計要素を 1 つずつ変えて段階表を作る）を、違う対象（ResNet ではなく DenseNet）に適用している。

| | **ConvNeXt**（2022） | **RDNet**（2024） |
|---|---|---|
| **出発点** | ResNet-50（76.1 → 78.8 でレシピ更新） | DenseNet-201（現代レシピで 79.7） |
| **到達点** | ConvNeXt-T **82.1** | RDNet-T **82.8** |
| **保持したもの** | 畳み込み（attention を使わない） | **連結ショートカット** |
| **輸入したもの** | Transformer の訓練レシピと設計 | ConvNeXt のブロック設計 + 訓練レシピ |
| **最大の単一要因** | **訓練レシピ**（総改善の約 46%） | **遷移層を増やす**（+1.5） |
| **主張** | ViT が CNN を置き換えたわけではない | **加算が連結を置き換えたわけではない** |

**RDNet は ConvNeXt のブロックをそのまま借りている**（LN・depthwise・7×7・活性化を減らす）。つまり**「ConvNeXt の成果を土台にして、ConvNeXt が触らなかった軸（ショートカットの型）を変えた」**という構図で、系譜として素直に接続する。

**[[concepts/convolutional-neural-network]] の「CNN は終わったのか」という問いに新しい材料を出す。** ConvNeXt V2 が MIM 互換性を解き、[[entities/nfnet]] が正規化を捨て、本論文がショートカットの型を問い直した。**CNN 側の設計空間はまだ枯れていない。**

**[[entities/swin-transformer]] / [[concepts/vision-transformer]] とも直接ぶつかる。** ViT も Swin も内部は加算ショートカットなので、本論文の主張が正しければ**Transformer 側にも連結ショートカットを試す余地がある**。論文はこれを今後の課題として明示していないが、Appendix 0.G で **WGAN の生成器を連結型に置き換えて FID 27.79 → 25.37** を示しており、**分類以外への一般化可能性**を示唆している。

## 用語と略称

- **RDNet** = Revitalized DenseNet（復活した DenseNet）。本論文のモデル。T/S/B/L の 4 スケール
- **DenseNet** = Densely Connected Convolutional Networks（Huang et al., CVPR 2017）
- **連結ショートカット（concatenation shortcut）** = $[\mathbf{X}, f(\mathbf{X}\mathbf{W})]$。チャネル方向に繋ぐ
- **加算ショートカット（additive shortcut）** = $\mathbf{X}+f(\mathbf{X}\mathbf{W})$。ResNet 以降の標準
- **GR（Growth Rate、成長率）** = 1 つの特徴混合器が出力し連結されるチャネル数。DenseNet の中心的なハイパーパラメータ
- **ER（Expansion Ratio、拡張率）** = inverted bottleneck の中間次元の倍率。**DenseNet は GR に掛けたが RDNet は入力次元に掛ける**
- **遷移層（transition layer）** = チャネル数を減らす層。RDNet では**ダウンサンプリングも兼ねる**
- **特徴混合器（feature mixer）** = RDNet の構築ブロック。出力が GR 次元
- **混合ブロック（mixing block）** = 3 つの特徴混合器 + 1 つの遷移層
- **特徴の再利用（feature reuse）** = 前の層の特徴をそのまま後段に渡すこと。DenseNet の中心概念
- **パッチ化ステム（patchification stem）** = 4×4 stride 4 の非重複畳み込み。ViT のパッチ分割と等価（[[concepts/vision-transformer]]）
- **チャネル再スケーリング** = channel layer-scale と squeeze-excitation を統合した機構
- **RandNet** = 本論文のパイロット研究で生成されるランダムネットワーク。空間 $\mathcal{A}$〜$\mathcal{E}$
- **PreNorm / PostNorm** = 正規化とショートカットの位置関係によるブロックの分類
- **CKA** = Centered Kernel Alignment。層の表現同士の類似度を測る指標
- **Stochastic Depth** = 訓練時に層を確率的にスキップする正則化
- **RSB-ResNet** = ResNet Strikes Back。現代的レシピで再訓練された ResNet
- **DeiT-III** = ViT の訓練レシピを改善した系譜。本論文の主要比較対象
- **HorNet / VAN / NAT / SMT / MogaNet / BiFormer / SLaK / RevCol** = 2022〜2024 年の最新アーキテクチャ群（本 wiki 未取り込み）。**InceptionNeXt は [[sources/inceptionnext]] として ingest 済み**（2026-09-01）
- **IS / FID** = Inception Score / Fréchet Inception Distance。生成モデルの評価指標
- **iNaturalist-2018/2019** = 長尾分布の細粒度分類ベンチマーク
- **ImageNet-V2 / -A / -R / -Sketch / ObjectNet** = 分布外（OOD）頑健性のベンチマーク

## 関連ページ

- [[translations/rdnet]] — 全文和訳（Appendix 0.A〜0.I 込み）
- [[entities/rdnet]] — モデルとしての RDNet
- [[concepts/skip-connection]] — 加算 vs 連結という軸の整理。本論文が受け皿になっている
- [[sources/convnext]] / [[entities/convnext]] — 対をなす論文。ブロック設計を借りている
- [[concepts/convolutional-neural-network]] — CNN の系譜と設計部品
- [[entities/swin-transformer]] / [[concepts/vision-transformer]] — 加算ショートカットを共有する相手
- [[entities/hrnet]] — 「特徴を捨てずに保持する」という別系統の設計
- [[entities/nfnet]] — 「定石を疑う」という同型の態度（正規化を捨てる）
