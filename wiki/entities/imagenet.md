---
type: entity
entity_kind: dataset
aliases: [ImageNet, ILSVRC, INet, IN-1k, ImageNet-1k]
tags: [dataset, benchmark, classification]
related: [[concepts/self-supervised-learning]], [[concepts/vision-transformer]]
sources: [[sources/dino-emerging-properties-in-self-supervised-vit]]
updated: 2026-05-24
---

# ImageNet

## 概要

**Computer Vision で最も歴史が長く影響力のある画像データセット**。Stanford の Fei-Fei Li らが 2009 年に公開し、その後の **ILSVRC（ImageNet Large Scale Visual Recognition Challenge, 2010〜2017）** という年次コンペティションを通じて、深層学習の爆発的発展を牽引した。

## バリエーション（用法が紛らわしいので要注意）

- **ImageNet（フル）/ ImageNet-22k / ImageNet-21k**: WordNet の名詞階層に基づく約 2.2 万カテゴリ、1400 万枚規模。**iBOT・DINOv2 はこの大きい方を事前学習で使う**（[[entities/dinov2]] の比較対象、また [[entities/lvd-142m]] のキュレーション「お手本」）。
- **ImageNet-1k / ILSVRC-2012**: 1000 カテゴリ、訓練 128 万枚、検証 5 万枚、テスト 10 万枚。**「ImageNet」と論文で言うときは通常これ**を指す。
- **ImageNet-V2**: 2019 年に作られた新しい検証集合。バイアス検証用。DINOv2 はこれで OpenCLIP に対し +1.1% 上回ることで汎化を実証。
- **ImageNet-R / -A / -Sketch / -C**: それぞれロバストネス評価用の派生（rendition, adversarial, sketch, corruption）。DINOv2 の ImageNet-A スコア（ViT-g/14 で 75.9%）は SSL 系で群を抜く。
- **ImageNet-ReaL**: 2020 年に作られた、ラベル誤りを修正した再注釈版。
- **ImageNet-100**: 1000 カテゴリから 100 だけ抜粋した小型版。SSL の素早い実験で使われる。

DINO/DINOv2 論文で単に "ImageNet" と書かれているのは原則 **ImageNet-1k**。ただし DINOv2 のアブレーション（[[sources/dinov2-learning-robust-visual-features-without-supervision]] §6.2）で「INet-22k」と明示されている場合は **ImageNet-22k** を指す。

## なぜ重要か

- **2012 年の AlexNet（Krizhevsky et al.）が ILSVRC で他手法を圧倒**したことで、深層学習が CV を席巻するきっかけとなった。
- ResNet、Inception、DenseNet、EfficientNet、ViT などほぼすべての画像分類モデルが ImageNet-1k で「事前学習 → 下流ファインチューン」のパイプラインで使われてきた。
- **「ImageNet で何 % か」が CV モデルの de facto ベンチマーク**として 10 年以上君臨している。

## 評価指標

- **Top-1 accuracy**: モデルの第 1 候補が正解と一致する率。最も標準的。
- **Top-5 accuracy**: モデルの上位 5 候補のいずれかが正解と一致する率。ILSVRC 初期によく使われた。
- DINO 論文では top-1 で報告。

## 主な使われ方（SSL 文脈）

1. **ラベルなし事前学習データ**: SSL 手法は ImageNet-1k の画像だけ使って学習（ラベルは捨てる）。
2. **線形評価ベンチマーク**: 凍結特徴量の上に線形分類器を載せ、ImageNet-1k のラベルで分類精度を測る。
3. **k-NN 評価ベンチマーク**: 凍結特徴量と訓練データの k 近傍で投票。詳細: [[concepts/knn-evaluation-protocol]]。
4. **転移学習のソース**: ImageNet 事前学習 → CIFAR/iNaturalist/Flowers などに転移。
5. **教師あり比較対象**: SSL 表現と教師あり表現を比較する基準として常に登場。

## 注意点・批判

- **データ自体のラベル誤り**が一定割合存在することが知られている（数 % 程度）。
- **人物カテゴリのバイアスや、ステレオタイプの問題**が指摘され、人物関連クラスは公開後に大幅削除された。
- **JFT-300M（Google 内部）や LAION（公開・10 億枚規模）** など、より大きなデータが登場し、ImageNet 単独学習の優位性は SSL 以降相対的に下がっている。

## 関連ページ

- [[sources/dino-emerging-properties-in-self-supervised-vit]]: ImageNet-1k で SSL の事前学習・線形評価・k-NN 評価を行う代表例
- [[concepts/self-supervised-learning]]
- [[concepts/knn-evaluation-protocol]]
