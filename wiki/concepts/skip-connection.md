---
type: concept
aliases: [Skip Connection, スキップ接続, ショートカット, Shortcut, 残差接続, Residual Connection, 加算ショートカット, 連結ショートカット, Concatenation]
tags: [architecture, building-block, residual-learning, dense-connection]
related: ["[[concepts/convolutional-neural-network]]", "[[concepts/vision-transformer]]", "[[concepts/object-detection]]", "[[entities/inceptionnext]]"]
sources: ["[[sources/rdnet]]", "[[sources/convnext]]", "[[sources/hrnet]]", "[[sources/inceptionnext]]"]
updated: 2026-09-01
---

# Skip Connection（スキップ接続 / ショートカット）

## 一言で

**ある層の入力を、後段の層の出力に直接届ける経路。** 何のために届けるか（勾配 / 特徴 / 解像度）と、**どう合流させるか（加算 / 連結 / 融合）**で設計が分かれる。「**加算 vs 連結**」という選択は 2015 年以降ほぼ加算に固定されてきたが、[[sources/rdnet]]（ECCV 2024）がそれを正面から問い直した。

> 本ページの一次情報は [[sources/rdnet]] / [[sources/convnext]] / [[sources/hrnet]] の 3 件である。これを超える記述には ⚠ を付けた。

## 何を運ぶための経路か

**「スキップ接続」は 1 つの概念ではなく、少なくとも 3 つの異なる目的が同じ名前で呼ばれている。**

| 目的 | 何が問題か | 代表 |
|---|---|---|
| **勾配を運ぶ** | 深くすると勾配が消える | **ResNet** の残差接続 ⚠ |
| **特徴を運ぶ** | 後段で失われた情報を再利用したい | **DenseNet** の密結合、**U-Net** の skip ⚠ |
| **解像度を運ぶ** | ダウンサンプルで空間情報が失われる | **U-Net** / Hourglass の encoder-decoder ⚠、**[[entities/hrnet\|HRNet]]** の多解像度融合 |

**[[sources/hrnet]] はこの 3 つ目を極端に押し進めた例である。** 「落としてから復元する」（U-Net 型の skip）でも「dilated conv で落とさない」でもなく、**$1/4$〜$1/32$ の 4 ストリームを並列に維持して双方向に融合し続ける**。融合回数が効くこと（1 回 70.8 → 8 回 73.4 AP）が実証されており、**skip を「たまに繋ぐもの」から「常時流し続けるもの」に変えた**と読める。

## 合流のさせ方 — 加算か、連結か

**これが [[sources/rdnet]] の主題である。**

| | **加算（additive）** | **連結（concatenation）** |
|---|---|---|
| **式** | $\mathbf{X}_{l+1}=\mathbf{X}_{l}+f(\mathbf{X}_{l}\mathbf{W})$ | $\mathbf{X}_{l+1}=[\mathbf{X}_{l},\ f(\mathbf{X}_{l}\mathbf{W})]$ |
| **代表** | ResNet → ResNeXt → EfficientNet → **[[entities/convnext\|ConvNeXt]]** → ViT → **[[entities/swin-transformer\|Swin]]** | DenseNet → **[[entities/rdnet\|RDNet]]**、U-Net 系の skip |
| **チャネル数** | 変わらない | **層ごとに増える** |
| **利点** | 恒等写像の微分が常に 1 → 勾配消失を回避。**モジュール化しやすい** | **特徴の再利用**、パラメータ効率、初期層への明示的な教師の伝播 |
| **難点** | 同じ空間に足し込むので表現が似通いやすい ⚠ | **メモリを食う**、幅のスケーリングが難しい |

> **本 wiki の系譜の大半は加算側にある。** ViT も Swin も ConvNeXt も、内部のショートカットはすべて加算である。つまり [[concepts/vision-transformer]] と [[concepts/convolutional-neural-network]] の対立軸の下では、**残差学習という同じ土台を全員が共有していた**。[[sources/rdnet]] はここを問い直す。

### 連結が強い理由 — ランクの議論

[[sources/rdnet]] の予想は 2 段構えである。

**(1) 連結はランクを増やす。** 層の出力 $f(\mathbf{X}\mathbf{W})$（$\mathbf{W}\in\mathbb{R}^{d_{in}\times d_{out}}$）について、$d_{in}$ がそれほど小さくなければ非線形性のために $\mathrm{rank}(f(\mathbf{X}\mathbf{W}))$ は $d_{out}$ に近づく。**$d_{out}>d_{in}$ の層は増加した表現能力を提供する**——連結はまさに出力次元を入力より大きくする操作である。

**(2) 中間の次元削減はランクを大きく損なわない。** $\mathbf{X}\mathbf{W}_{1}f(\mathbf{W}_{2})$（$d_r<d_{in},d_{out}$）でも $d_r$ がそれほど小さくなければランクは保たれる。**したがって遷移層を頻繁に挟んでメモリを抑えてもよい。**

**この 2 つが「連結の利点を保ちつつメモリ問題を解く」という設計に直結する。**

### 検証 — 15,000 個のランダムネットワーク

[[sources/rdnet]] の検証設計が本主題にとって最も価値がある。

**Tiny-ImageNet 上で 15,000 個を超えるランダムネットワークをサンプルし、ショートカットだけを加算/連結で入れ替えて比較する。** パラメータ数・FLOPs・メモリを揃え、深さ・幅・活性化（5 種）・正規化（BN/LN）・カーネルサイズ（4 種）・ブロック型（3 種）・データ拡張・オプティマイザまで振る。

| 空間 | add | **concat** |
|---|---|---|
| $\mathcal{A}$（2M） | 45.8±2.0 | **47.5±2.1** |
| $\mathcal{B}$（4M） | 50.1±2.1 | **51.2±2.1** |
| $\mathcal{C}$（9M） | 53.2±2.2 | **54.3±2.0** |
| $\mathcal{D}$（+ 拡張） | 57.4±1.5 | **58.1±1.4** |
| $\mathcal{E}$（+ AdamW） | 58.2±1.5 | **58.9±1.6** |

**全 10 設定で連結が勝つ。** ImageNet-1K へスケールアップした追加実験（Appendix 0.I.2）でも同じ傾向が出る。

> **この検証設計は「設計空間を population として評価する」という RegNet 由来の手法である** ⚠。単一のモデルを 2 つ作って比べるのではなく、**設計空間全体から無作為抽出した分布同士を比べる**ことで「たまたまチューニングが上手くいっただけ」という交絡を潰す。[[sources/l2rw]] が「乱数ベースラインを置け」と警告した問題への、別方向からの回答とも読める。

**ただし差は小さい。** 平均差は約 1 ポイントで標準偏差（約 2）より小さい。しかも [[sources/rdnet]] は自ら次を記録している。

> **データ拡張が加算的ショートカットと連結的ショートカットの間の性能差を縮める。特に AdamW で stochastic depth を用いて訓練された要素は加算的ショートカットからより恩恵を受ける。**

実際 $\mathcal{E}$ 空間の Stochastic Depth（57.5 vs 57.5）と RandErase（58.1 vs 58.1）では**差が消える**。**「連結が本質的に強い」という主張と「近代的レシピが差を埋める」という自身の発見は緊張関係にある。**

### 連結型の代償と、その解き方

連結の最大の難点は**メモリと幅のスケーリング**である。DenseNet-161/-233 のような広いモデルはメモリを大量に消費し、これが DenseNet の衰退の直接の原因になった。

[[sources/rdnet]] の解は**遷移層を増やすこと**だった。各ステージの後だけでなく **3 ブロックごとにストライド 1 の遷移層を挟んでチャネルを圧縮する**。近代化ロードマップで**最大の跳ね（+1.5%p）**を生んだのがこのステップである。

**それでも ResNet 系には負ける**: RDNet-T のメモリ 4.1GB に対し ConvNeXt-T 2.7GB、Swin-T 2.6GB。**連結の代償は緩和できても消えない。**

## 表現への影響 — CKA が示すもの

[[sources/rdnet]] は **CKA（Centered Kernel Alignment）**で層ごとの表現の類似度を測っている。

<figure>

![](../../raw/assets/rdnet/fig5.png)

<figcaption>図（[[sources/rdnet]] 図6 より再掲）: 左が RDNet-T の層間類似度、中央が ConvNeXt-T、右が両者の間。RDNet は対角以外が全体に暗く各層で異なる特徴を学習している一方、ConvNeXt は中盤の層同士が明るく似た特徴を繰り返し学習している。</figcaption>
</figure>

**加算は同じ空間に足し込むので表現が似通いやすく、連結は新しいチャネルを足すので分化しやすい**、という読み方ができる ⚠（論文はこの解釈を明示していない）。ただし**多様性が高いこと自体がなぜ良いのかは論じられていない**点は注意が要る。

## 「skip があるから要らない」という定石を疑う

**スキップ接続の存在が、他の設計要素の要否を左右する**という論点が繰り返し現れる。

| 定石 | 誰が疑ったか | 結果 |
|---|---|---|
| **DenseNet は接続パターンが似ているので Stochastic Depth は不要** | [[sources/rdnet]] | **0 → 0.15 で +1.2%p**。ロードマップのどの単一ステップより大きい |
| **残差接続があるので正規化層は必須** | [[entities/nfnet\|NFNet]]（[[sources/nfnet]]） | **正規化を完全に排除**しても、むしろ速く強くなる |
| **ViT の特殊モジュール（shifted window 等）は必須** | [[entities/hiera]] | MIM で適切に事前学習すれば**全部要らない** |

**いずれも「アーキテクチャの一部が別の一部の要否を暗黙に決めている」という思い込みを解いた例**である。

## 第 3 の軸: ショートカットの「本数」

加算か連結かとは別に、**1 ブロックにショートカットを何本置くか**という軸がある。[[entities/inceptionnext|InceptionNeXt]]（[[sources/inceptionnext]], CVPR 2024）がこれを明示的に扱った。

Transformer 系のブロック（**MetaFormer** の枠組みで抽象化されるもの）は**ショートカットを 2 本**持つ——トークン混合器（空間を混ぜる部品）用と MLP（チャネルを混ぜる部品）用である。ところが ConvNeXt のブロックは**1 本しか持たない**。InceptionNeXt はこの 1 本版を **MetaNeXt** と名付けて抽象化した。

| | **MetaFormer** | **MetaNeXt** |
|---|---|---|
| ショートカット | **2 本** | **1 本**（2 つの残差サブブロックを統合） |
| 構造 | Norm → Mixer → (+) → Norm → MLP → (+) | Mixer → Norm → MLP → (+) |
| 具体例 | ViT / Swin / PoolFormer | **ConvNeXt** / InceptionNeXt |

**そして「1 本に減らす」ことには代償があった。** 著者は MetaNeXt のトークン混合器を自己注意にした **MetaNeXt-Attn** を訓練し、**top-1 が 3.9%——つまり収束しない**ことを報告している（同条件の DeiT-S は 79.8、ConvNeXt-S 等方版は 79.7）。

> **論文の結論は「MetaNeXt ブロックのトークン混合器は複雑すぎてはならない」。** ショートカットを 2 本持てば複雑な混合器を置けるが、1 本に減らすと**混合器の選択肢そのものが狭まる**。

**これは本節の主題——「アーキテクチャの一部が別の一部の要否を暗黙に決めている」——の、最も直接的な実証である。** 上の表の 3 例が「skip があるから X は要らない、を疑う」形だったのに対し、これは**「skip を減らすと X が置けなくなる」**という逆向きの依存関係を示している。

**含意**: ConvNeXt がショートカット 1 本で成立しているのは、**トークン混合器が depthwise 畳み込みという単純な部品だから**であって、一般に成り立つ簡素化ではない。⚠ ただしこれは 1 論文の 1 実験（等方的アーキテクチャ、22M、IN-1K）に基づく報告であり、失敗の原因（勾配伝播か、正規化の位置か、学習率か）は論文でも分析されていない。

## 現在地

本 wiki に取り込まれた範囲では:

1. **加算が事実上の標準であり続けている**。ViT / Swin / ConvNeXt / MAE すべて加算
2. **連結は密予測と特徴再利用の文脈で生き残った**。U-Net 系の skip、[[entities/hrnet]] の多解像度融合、[[entities/rdnet]]
3. **[[sources/rdnet]] が「加算 vs 連結」を初めて系統的に比較した**。連結が一貫して勝つが差は小さく、近代的レシピで縮む
4. **本数という軸は 1 論文しか触れていない** ⚠。[[sources/inceptionnext]] の MetaNeXt-Attn（2 本 → 1 本にすると自己注意が置けなくなる）が唯一の一次情報
5. **Transformer 側での連結ショートカットは未検証** ⚠。[[sources/rdnet]] は CNN のみを扱い、この方向を今後の課題として明示もしていない。ただし Appendix 0.G で **WGAN の生成器を連結型に置き換えて FID 27.79 → 25.37** を示しており、分類以外への一般化可能性は示唆されている

## 関連ページ

- [[sources/rdnet]] / [[entities/rdnet]] — 「加算 vs 連結」を系統的に比較した唯一の一次情報
- [[sources/hrnet]] / [[entities/hrnet]] — 解像度を運ぶ skip の極端な形（並列維持 + 反復融合）
- [[sources/convnext]] / [[entities/convnext]] — 加算側の代表。RDNet がブロック設計を借りた相手
- [[concepts/convolutional-neural-network]] — CNN の系譜と設計部品
- [[concepts/vision-transformer]] — ViT も内部は加算ショートカット
- [[entities/nfnet]] — 「skip があるから正規化が要る」という定石を疑った例
- [[entities/hiera]] — 「事前学習が適切なら特殊モジュールは要らない」という同型の議論
- [[entities/inceptionnext]] / [[sources/inceptionnext]] — ショートカットの「本数」という第 3 の軸（MetaFormer 2 本 vs MetaNeXt 1 本）

## 今後の ingest 候補

⚠ 以下は本 wiki 未取り込みで、本ページの記述を一次情報に格上げするために有用。

- **ResNet**（He et al., CVPR 2016） — 加算ショートカットの原典。**本 wiki に依然として専用ページがない**
- **Identity Mappings in Deep Residual Networks**（He et al., ECCV 2016） — pre-activation と恒等写像の解析
- **DenseNet**（Huang et al., CVPR 2017） — 連結ショートカットの原典
- **U-Net**（Ronneberger et al., MICCAI 2015） — encoder-decoder の skip の原典
- **Deep Networks with Stochastic Depth**（Huang et al., ECCV 2016）
- **RegNet**（Radosavovic et al., CVPR 2020） — 「設計空間を population として評価する」手法の原典
- **MetaFormer Is Actually What You Need for Vision**（Yu et al., CVPR 2022） — ショートカット 2 本の抽象ブロックの原典。[[sources/inceptionnext]] の MetaNeXt はこれの単純化版
