---
type: entity
entity_kind: dataset
aliases: [LVD-142M, Large-scale Visual Dataset 142M]
tags: [dataset, ssl, curated, meta-ai]
related: [[concepts/self-supervised-learning]], [[concepts/foundation-model]]
sources: [[sources/dinov2-learning-robust-visual-features-without-supervision]]
updated: 2026-05-24
---

# LVD-142M

## 概要

**LVD** = **Large-scale Visual Dataset**。Meta AI が DINOv2（[[entities/dinov2]]）の事前学習のために構築した **142M 枚（1.42 億枚）** の画像コレクション。**自動キュレーションパイプライン**を経て、Web からクロールした 12 億枚の生画像から選別された。

- 出典: [[sources/dinov2-learning-robust-visual-features-without-supervision]] §3
- 公開状況: **非公開**（収集パイプラインのコードのみ <https://github.com/facebookresearch/dinov2> で公開）

## なぜ重要か

それまでの SSL は ImageNet-1k（120 万枚）か ImageNet-22k（1400 万枚）で学習するのが標準だった。LVD-142M は：

- **規模**: ImageNet-22k の **10 倍**、ImageNet-1k の **120 倍**
- **多様性**: Web 由来なので、ImageNet にないドメイン（自然画像以外）も含む
- **キュレーション済み**: ただし Web 未加工データそのままではなく、自動パイプラインで品質と多様性を担保

この組み合わせ（**大規模 + 多様 + キュレーション**）が、DINOv2 が「弱教師あり SOTA に追いつく」鍵だった。

## 自動キュレーションパイプライン（§3）

LVD-142M の最大の貢献は、**テキストもメタデータも一切使わず、視覚的類似度だけで自動キュレーション**するパイプラインの提案。

### Step 1: データソース

- **「お手本」となるキュレーション済みデータ**:
  - ImageNet-22k
  - ImageNet-1k 訓練分割
  - Google Landmarks v2（GLDv2）
  - 細粒度データセット群（Cars, Aircraft, Flowers など）
  - → 詳細は論文付録 A の表 15

- **「素材」となる未キュレーションデータ**:
  - 公開 Web クロールデータ（リポジトリ名は論文中では明示されず）
  - 各 Web ページの `<img>` タグから URL 抽出
  - 安全性チェック（unsafe / restricted URL を除外、NSFW フィルタ、顔ぼかし）
  - 結果: 12 億枚の生画像

### Step 2: 重複排除

- [88] のコピー検出パイプラインを適用
- **近重複画像を除去**して冗長性を減らし、多様性を確保
- ベンチマーク（テスト / 検証集合）に含まれる画像との近重複も除去
  - → ベンチマーク汚染（contamination）の防止

### Step 3: 埋め込み計算

- **ImageNet-22k で事前学習した自己教師あり ViT-H/16** で全画像を埋め込み化
- 距離はコサイン類似度

### Step 4: 検索ベースのキュレーション

- 未キュレーションデータを k-means クラスタリング
- 各「お手本」クエリ画像に対して未キュレーションデータから **N=4 個の最近傍**を取得
- お手本データセットが小さい場合は、対応するクラスタから M 個サンプリング

> **補足: なぜ N=4 か** — 視覚検査では N をもっと大きくしても品質は良い。しかし N を大きくすると **「同じ未キュレーション画像が複数のクエリの最近傍として選ばれる」衝突（collision）**が増え、データの実効的な多様性が下がる。N=4 がそのトレードオフの良い点だったと著者らは報告（§3）。

### Step 5: 実装

- **Faiss ライブラリ**で GPU 加速の近似最近傍検索
- inverted file index + product quantization で大規模ベクトル検索を効率化
- 20 ノード × 8 GPU（V100-32GB）で 2 日未満で完成

## ImageNet-22k との比較（§6.2）

DINOv2 著者らが行ったアブレーション（同じ iterations で ViT-g/14 を訓練）：

| 訓練データ | INet-1k | iNat 2021 | Oxford-M | Places205 |
|---|---|---|---|---|
| ImageNet-22k | 85.9 | 85.6 | 62.5 | 67.0 |
| INet-22k \ INet-1k | 85.3 | 85.1 | 58.7 | 66.5 |
| 未キュレーション 142M | 83.3 | 76.4 | 54.3 | 67.2 |
| **LVD-142M** | **85.8** | **86.4** | **64.6** | **67.6** |

ポイント：
- **キュレーションすると未キュレーションより大きく改善**（ImageNet-1k: 83.3 → 85.8、Oxford-M: 54.3 → 64.6）
- **ImageNet-22k と同等以上**（ImageNet-1k は若干劣るが、他は上回る）
- **キュレーションプロセスで使ってない領域（iNat, Places205）でも改善** → スケールと多様性が未見ドメインにも効く

## 公開性と再現性

LVD-142M 自体は**公開されていない**。理由は明示されていないが、推測される要因：

- Web クロールデータの再配布に伴う著作権・プライバシー問題
- 画像 URL リスト公開でも、リンク切れで再現困難
- Meta 内部の競争優位性

代替として、**LAION-2B / LAION-5B**（OpenCLIP の訓練データ）や **DataComp** が公開大規模画像データセットとして使える。ただしいずれも画像 - テキスト対で、画像のみの大規模キュレーション集合は LVD-142M ほどの規模で公開されたものはまだ少ない。

DINOv2 のコードは公開されているので、原理的には独自データで類似パイプラインを動かせる。

---

## 後継

**LVD-1689M**（[[entities/lvd-1689m]], 2025）が DINOv3（[[entities/dinov3]]）用に構築された：

- **規模 12 倍**: 142M → 1.689B
- **データソース変更**: 公開 Web クロール → Instagram 公開投稿
- **キュレーション手法変更**: retrieval ベース → **階層 k-means**（[196]）
- **画像埋め込み器**: ImageNet-22k pretrained ViT-H/16 → **DINOv2 自身**

「前世代モデル（DINOv2）で次世代データ（LVD-1689M）をキュレートする」という循環構造が確立された。

## 注意点

- **Web 由来データの法的・倫理的問題**: 著作権・プライバシー・偏り。論文 §8.1 で地理的バイアス（西洋・高所得地域偏重）を著者ら自身が認めている。
- **画像内テキスト**: Web 画像にはスクリーンショットや figure など「画像内に文字を含むもの」が多く含まれる可能性。これが特徴量に与える影響は議論の余地あり。

## 関連ページ

- [[sources/dinov2-learning-robust-visual-features-without-supervision]]: LVD-142M を構築・利用した論文
- [[entities/dinov2]]: LVD-142M で学習されたモデル
- [[entities/imagenet]]: キュレーションの「お手本」の一つ
- [[concepts/self-supervised-learning]] / [[concepts/foundation-model]]
