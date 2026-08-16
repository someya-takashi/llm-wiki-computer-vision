---
type: concept
aliases: [CNN, ConvNet, Convolutional Neural Network, 畳み込みニューラルネットワーク]
tags: [architecture, cnn, backbone]
related: ["[[vision-transformer]]", "[[masked-image-modeling]]", "[[object-detection]]", "[[self-supervised-learning]]"]
sources: ["[[sources/convnext]]", "[[sources/convnext-v2]]", "[[sources/vision-transformer]]"]
updated: 2026-06-17
---

# Convolutional Neural Network（CNN / ConvNet, 畳み込みニューラルネットワーク）

## 一言で

**画像を小さなフィルタ（カーネル）で端から端まで走査し、局所的なパターンを検出する層を積み重ねたネットワーク**。2012 年の AlexNet から 2020 年の ViT 登場まで、Computer Vision の圧倒的な主役だった。「CNN」と「ConvNet」は同義で、論文により呼称が異なる（本 wiki では両方を使う）。

> **用語注**: 本 wiki では `CNN` と `ConvNet` を同じものとして扱う。ConvNeXt 論文（[[sources/convnext]]）は一貫して "ConvNet" を用いる。

## なぜ CNN は画像に効くのか — 3 つの構造的前提

CNN の強さは「畳み込み」という演算が**画像についての仮定を構造として埋め込んでいる**ことにある。これを**帰納バイアス（inductive bias）**と呼ぶ。

1. **局所性（locality）**: 近い画素どうしが関係する、という仮定。カーネルは 3×3 や 7×7 といった狭い範囲しか見ない。
2. **重み共有（weight sharing）**: 同じフィルタを画像全体で使い回す。「左上の犬の耳」と「右下の犬の耳」を別々に学ぶ必要がない。パラメータ数も劇的に減る。
3. **平行移動同変性（translation equivariance）**: 入力の物体がずれたら、出力の特徴マップも同じだけずれる。

> **補足: 同変性（equivariance）と不変性（invariance）は別物** — よく混同されるが区別が重要。
> - **同変性**: 入力がずれる → 出力も**同じだけずれる**。畳み込み演算そのものが持つ性質。「どこに何があるか」の情報が保たれるので、**検出やセグメンテーションのような位置が問われるタスクに有用**（[[sources/convnext]] §1 が最重要の帰納バイアスとして挙げるのはこちら）。
> - **不変性**: 入力がずれても出力は**変わらない**。プーリングや最終段の大域平均プーリングを重ねることで**近似的に**得られる性質。分類のように「どこにあるか」を捨てたいタスクで有用。
>
> つまり CNN は「畳み込み層で同変性を保ちながら、プーリングを重ねて徐々に不変性へ寄せていく」構造になっている。[[vision-transformer]] の「CNN との比較」表では簡略化のため「平行移動不変性」と書いているが、より正確には上記の使い分けになる。

さらに実務上重要なのが **計算共有** である。スライディングウィンドウ的に使うとき、重なり合う領域の計算が共有されるため、高解像度画像でも効率的に処理できる。ここが self-attention が入力サイズの二次で重くなる [[vision-transformer]] との決定的な差になる。

## アーキテクチャの系譜

| 年 | モデル | 主な貢献 |
|---|---|---|
| 1989/1998 | LeNet | 誤差逆伝播で訓練される ConvNet の原型（手書き数字認識） |
| 2012 | **AlexNet** | ImageNet で圧勝し「ImageNet の瞬間」を起こす。深層学習革命の起点 |
| 2014 | VGGNet | **3×3 の小カーネルを積み重ねる**という長年の標準を確立 |
| 2014 | GoogLeNet / Inception | 複数カーネルサイズの並列分岐、1×1 conv によるチャネル削減 |
| 2015 | **ResNet** | **残差接続（skip connection）**で超深層の訓練を可能に。以後すべての基準点 |
| 2016 | DenseNet | 全層間の密な接続 |
| 2017 | **ResNeXt** | **grouped convolution** +「グループを増やして幅を拡げる」原則 |
| 2017 | MobileNet / Xception | **depthwise separable convolution** を普及させる |
| 2018 | **MobileNetV2** | **inverted bottleneck** を普及させる |
| 2019 | EfficientNet | 深さ・幅・解像度の複合スケーリング則 |
| 2020 | RegNet | 設計空間そのものを探索する方法論 |
| 2022 | **ConvNeXt**（[[entities/convnext]]） | Transformer 流の訓練レシピと設計を取り込み、Swin を上回る |
| 2023 | **ConvNeXt V2**（[[entities/convnext-v2]]） | **FCMAE（疎畳み込み MIM）+ GRN** と共設計。CNN でも MIM が効くことを実証、IN-1K 88.9% |

## 主要な構成部品

以下の各項目には、[[sources/convnext]] の近代化アブレーション（ResNet-50 → ConvNeXt-T、ImageNet-1K top-1）が実証データを与えている。**これは同一の訓練レシピ下で 1 要素ずつ変えた貴重な対照実験**である。

### stem（入口）

画像を最初に受け取り、一気にダウンサンプリングする層。

- **ResNet 流**: 7×7・stride 2 の畳み込み + max pool（合計 4× 縮小）
- **patchify 流**: 4×4・stride 4 の**非重複**畳み込み（ConvNeXt / Swin）。ViT のパッチ分割（[[vision-transformer]]）は、まさに stride $N$ の畳み込み 1 層と等価であり、**「ViT の patchify」と「大カーネル非重複 conv の stem」は同じもの**
- 実験: 置換しても 79.4 → 79.5 と**ほぼ変わらない**。stem の設計は性能の主因ではない

### stage 構成（解像度の段階）

解像度を半分ずつ落としながら 4 段階に分ける階層構造。この階層性こそが密予測タスク（検出・セグメンテーション）で CNN が有利だった理由であり、Swin Transformer が Transformer 側に輸入した性質でもある。

- 各段階へのブロック配分（stage compute ratio）は ResNet-50 で 3:4:6:3、Swin-T / ConvNeXt-T では **3:3:9:3**
- 実験: 配分変更で 78.8 → 79.4（+0.6）

### bottleneck と inverted bottleneck

- **bottleneck**（ResNet）: `1×1 で細く → 3×3 → 1×1 で戻す`。中間が**細い**
- **inverted bottleneck**（MobileNetV2 以降）: `1×1 で太く → depthwise → 1×1 で戻す`。中間が**太い**（通常 4 倍）
- **Transformer の MLP ブロックは、実は inverted bottleneck と同じ形**（隠れ次元が入力の 4 倍）。ConvNeXt はこの一致を明示的に指摘した
- 実験: 80.5 → 80.6（ResNet-50 領域では +0.14 と小さいが、**ResNet-200 領域では +0.79 と大きい**。効き方はモデル規模に依存する）

### grouped / depthwise convolution

- **grouped convolution**（ResNeXt）: フィルタをグループに分けて適用。FLOPs を減らせるので、浮いた分で幅を拡げる
- **depthwise convolution**: グループ数 = チャネル数の極端な場合。**チャネル間を混ぜず、空間方向だけ混合する**
- **なぜ Transformer 的なのか**: depthwise conv（空間だけ混ぜる）+ 1×1 conv（チャネルだけ混ぜる）という分離は、self-attention（空間だけ混ぜる）+ MLP（チャネルだけ混ぜる）という Transformer の構造と**同じ役割分担**になっている
- 実験: depthwise 化だけだと 80.5 → 78.3 と**下がる**（FLOPs も半減）。幅を 64 → 96 に拡げて初めて 80.5 に戻る

### カーネルサイズ

- VGGNet 以来「3×3 を積む」が標準だった。大カーネルは GPU 実装効率が悪いとされてきた
- ConvNeXt の再検証: 3 → 5 → **7** で改善し、**7×7 で飽和**（9, 11 では改善しない）。Swin の窓サイズ 7×7 と一致するのは示唆的
- ただし ResNet-200 領域では **5 で飽和**する

### 正規化: BatchNorm vs LayerNorm

- **BatchNorm（BN）**: ミニバッチ方向に正規化。ConvNet の事実上の標準で、収束を速め過学習を減らす。ただし**バッチサイズ依存**、訓練/推論で挙動が変わる、分散学習で扱いが面倒、といった厄介さがある
- **LayerNorm（LN）**: 特徴次元方向に正規化。バッチに依存しない。Transformer の標準
- 素の ResNet で BN を LN に置き換えると性能が落ちるが、**他の近代化をすべて済ませた後なら LN の方がわずかに良い**（81.41 → 81.47）
- **落とし穴**: BN を持つモデルでは **EMA（重みの指数移動平均）が性能を著しく害する**（[[sources/convnext]] Appendix A.1）

### 活性化関数と「部品を減らす」という発想

ConvNeXt の近代化で**終盤に最も効いたのは、部品を足すことではなく減らすこと**だった。

- **ReLU → GELU**: +0.05 と**ほぼ無効**。「Transformer で使われている部品を入れれば良くなる」わけではない
- **活性化を減らす**（ブロック内の GELU を 1 個だけに）: **+0.65** で Swin-T に並ぶ
- **正規化を減らす**（BN を 1 個だけに）: +0.14 で Swin-T を超える

結果として ConvNeXt ブロックは ResNet ブロックより部品が少ない（BN×3 + ReLU×3 → LN×1 + GELU×1）。

## 「CNN は終わったのか」— 現在地

1. **2020: ViT が分類で ConvNet を抜く**（[[sources/vision-transformer]]）。ただし大規模事前学習が前提で、二次計算量ゆえ高解像度タスクに弱い
2. **2021: Swin Transformer が ConvNet の性質を輸入して汎用バックボーン化**。局所窓 attention + 階層構造という、まさに ConvNet が元から持っていたもの
3. **2022: ConvNeXt が反論**（[[sources/convnext]]）。訓練レシピと設計を揃えれば純粋 ConvNet が Swin を上回る。**「性能差の最大の単一要因は訓練レシピ（+2.7）であり、比較されていた ResNet が 2015 年のレシピで訓練された古い数字だった」**
4. **2023: ConvNeXt V2 が「CNN は MIM に向かない」という残った弱点も潰す**（[[sources/convnext-v2]]）。**疎畳み込みでマスク領域からの情報漏洩を遮断する FCMAE** と、特徴崩壊を防ぐ **GRN 層**を*共設計*することで、CNN でもマスク画像モデリングが効くことを実証（IN-1K **88.9%**、公開データのみで当時 SOTA）。3.7M の Atto から 659M の Huge まで全サイズで教師ありを +0.8〜+1.5 上回った
5. **それでも主流は ViT に収束した**。CLIP / SigLIP / DINOv2 / DINOv3 / MLLM の視覚エンコーダはほぼ例外なく ViT。**V2 以後に残る決定的な理由は「言語との接続性」**である:
   - **パッチ列という表現が言語トークンと素直に接続できる**（[[weakly-supervised-pretraining]] / MLLM）。**ConvNeXt V2 もこの軸には手をつけていない**
   - 可変解像度・可変トークン数の扱いが柔軟（[[questions/vit-dynamic-resolution-evolution]]）
   - なお **MIM 適性は V2 で解消済み**なので、「CNN は MIM に向かない」を現在の理由として挙げるのは正確ではない
6. **CNN は「効率が要る場所」で生き残った**。[[entities/dinov3]]（2025）は ViT-7B から **ConvNeXt-T/S/B/L（V1 系）を蒸留**し、量子化・エッジ展開向けに配布している

> **要するに** — 「CNN が ViT に負けた」のではなく、「**分類・検出・セグメンテーションでは互角で、自己教師あり事前学習も（共設計すれば）効くが、言語との接続性で ViT が選ばれた**」というのが実態に近い。ConvNeXt V1 が前半を、V2 が中盤を示し、残ったのが最後の 1 点、という整理ができる。

## 関連ページ

- [[sources/convnext]] / [[entities/convnext]] — CNN を Transformer 流に近代化した論文。本ページの主要な実証データ源
- [[sources/convnext-v2]] / [[entities/convnext-v2]] — CNN と自己教師あり学習を共設計した続編（FCMAE + GRN）
- [[vision-transformer]] — 対比されるアーキテクチャ。「CNN との比較」節と対で読む
- [[sources/vision-transformer]] — ViT 原典。「大規模訓練は帰納バイアスに勝る」という主張の出所
- [[masked-image-modeling]] — かつて CNN の弱点だった領域。ConvNeXt V2 が疎畳み込みで解決した
- [[object-detection]] — R-CNN ファミリー / YOLO など CNN 時代の検出器の系譜
- [[entities/hiera]] — 階層型 ViT から特殊モジュールを削ぐ研究。ConvNeXt と同じ *simplicity* 論の Transformer 側
- [[entities/dinov3]] — ConvNeXt バリアントを蒸留で提供する後年の基盤モデル
