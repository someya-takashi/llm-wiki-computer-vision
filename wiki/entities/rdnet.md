---
type: entity
entity_kind: model
aliases: [RDNet, RDNet-T, RDNet-S, RDNet-B, RDNet-L, Revitalized DenseNet, DenseNets Reloaded]
tags: [rdnet, densenet, concatenation, skip-connection, convolutional-neural-network, backbone, naver, eccv2024]
related: ["[[concepts/skip-connection]]", "[[entities/convnext]]", "[[entities/swin-transformer]]", "[[concepts/convolutional-neural-network]]", "[[entities/hrnet]]", "[[entities/nfnet]]", "[[entities/inceptionnext]]"]
sources: ["[[sources/rdnet]]"]
updated: 2026-09-01
---

# RDNet（Revitalized DenseNet）

NAVER Cloud AI / NAVER AI Lab の **Donghyun Kim, Byeongho Heo, Dongyoon Han** による、**DenseNet を近代化した CNN バックボーン**（arXiv:2403.19588, **ECCV 2024**）。詳細解説: [[sources/rdnet]] / 翻訳: [[translations/rdnet]]。

## 一言で

**[[entities/convnext|ConvNeXt]] が ResNet に対してやったことを DenseNet に対してやった。** ただし主張はもう一段深く、**「加算ショートカットより連結ショートカットの方が本来強い」**という命題を **15,000 個のランダムネットワーク**で検証したうえで、Swin・ConvNeXt・DeiT-III を上回るモデルを組み立てている。

## アーキテクチャ

<figure>

![](../../raw/assets/rdnet/fig1.png)

<figcaption>図1（再掲）: RDNet の模式図。4 ステージ構成で、各 Stage-N は L_N 個の混合ブロックからなる。混合ブロックは 3 つの特徴混合器 f と 1 つの遷移層で構成される（最後の混合ブロックは遷移層を持たない）。丸 C が連結を表す。</figcaption>
</figure>

```
Stem（パッチ化 4×4 s4 + LN）
  → Stage-1 → Transition S/2 → Stage-2 → Transition S/2
  → Stage-3 → Transition S/2 → Stage-4
  → GlobalAvgPool → LayerNorm → FC
```

| 部品 | 構成 |
|---|---|
| **特徴混合器 $f$** | `DWConv 7×7 (C→C)` → `LayerNorm` → `Linear (C→4C)` → `GELU` → `Linear (4C→GR)` |
| **混合ブロック** | 特徴混合器 ×3 + 遷移層 ×1（最後のブロックは遷移層なし） |
| **遷移層** | `LayerNorm` → `Conv s×s stride s (C→C/2)`。**次元削減とダウンサンプリングを兼ねる** |
| **ショートカット** | **連結**（加算ではない） |

**設計の肝は「特徴混合器の出力が GR 次元である」こと。** ConvNeXt のブロックが入力と同じ次元を返して加算するのに対し、**RDNet のブロックは GR 次元だけ返して連結される**。

## バリアント

| モデル | GR（成長率） | B（特徴混合器の数） | Param | FLOPs | Top-1 |
|---|---|---|---|---|---|
| **RDNet-T** | (64, 104, 128, 224) | (3, 3, 12, 3) | 24M | 5.0G | **82.8** |
| **RDNet-S** | (64, 128, 128, 240) | (3, 3, 21, 6) | 50M | 8.7G | **83.7** |
| **RDNet-B** | (96, 128, 168, 336) | (3, 3, 21, 6) | 87M | 15.4G | **84.4** |
| **RDNet-L** | (128, 192, 256, 360) | (3, 3, 24, 6) | 186M | 34.7G | **84.8** |
| RDNet-L (384²) | 同上 | 同上 | 186M | 101.9G | **85.8** |

**特徴混合器の数は 3 の倍数**（$B_N = 3L_N$、$L_N$ は混合ブロック数）。**GR はステージごとに異なる**——一様な GR は精度と効率の両方を損なう（原典 表8d）。

## 近代化ロードマップ

**[[sources/convnext]] と同じ形式の段階表**（原典 表1）。

| | 要素 | Top-1 | Lat b1 (ms) | Mem (GB) |
|---|---|---|---|---|
| (a) | DenseNet-201（現代レシピで再訓練） | 79.7 | **38.4** | 3.9 |
| (b) | + **より広く浅く**（GR 32 → 120） | 79.5 (−0.2) | **8.5 (−29.9)** | 3.2 |
| (c) | + 近代化されたブロック（LN / depthwise / 7×7） | 80.4 (+0.9) | 10.4 | 3.4 |
| (d) | + 中間チャネル次元↑（**ER を GR から切り離す**） | 80.8 (+0.4) | 11.8 | 3.1 |
| (e) | + **遷移層↑（3 ブロックごと）** | **82.3 (+1.5)** | 11.0 | 3.4 |
| (f) | + パッチ化ステム | 82.4 (+0.1) | 11.0 | 3.2 |
| (g) | + 洗練された遷移層 | 82.6 (+0.2) | 13.6 | 3.1 |
| (h) | + チャネル再スケーリング | **82.8 (+0.2)** | 14.0 | 3.1 |

- **(b) は精度を下げるがレイテンシを 4.5 倍速くする**——速度のための変更
- **最大の跳ね (+1.5) は「遷移層を増やす」**。連結で膨れる特徴を 3 ブロックごとに圧縮することで高い GR を維持できる
- **(d) が DenseNet との決定的な違い**。DenseNet は ER を GR に掛けた（ER = 4 × GR）が、RDNet は**入力次元に掛ける**

## 主要結果

| タスク | RDNet | ConvNeXt | Swin | DeiT-III |
|---|---|---|---|---|
| ImageNet-1K（-T） | **82.8** | 82.1 | 81.3 | 81.4（-S） |
| ImageNet-1K（-B） | **84.4** | 83.8 | 83.5 | 83.8 |
| ImageNet-1K（-L, 384²） | **85.8**（186M） | 85.5（198M） | – | 85.8（**304M**） |
| ADE20K mIoU ss（-B） | **49.6** | 49.1 | 48.1 | 49.3 |
| COCO AP^box（-T, Mask-RCNN 3x） | **47.5**（43M） | 46.2（48M） | 46.0（48M） | – |
| ゼロショット CLIP（-B） | **54.1** | 51.2 | – | – |

**バッチサイズ 1 のレイテンシが際立つ**: RDNet-T **7.4ms** に対し HorNet-T 21.2ms、SMT-S 46.9ms、BiFormer-S 51.6ms（**7 倍差**）。RDNet-B の 11.7ms は、より小さい MogaNet-S（20.0ms）より速い。**ただし b128 では差が縮む**（RDNet-T 175.2 vs HorNet-T 183.7）。

### 転移性が最も驚く（Appendix 0.F）

| Model | Param | iNat18 | iNat19 |
|---|---|---|---|
| DeiT-III-S | 22 | 67.1 | 72.7 |
| **RDNet-T** | **24** | **77.0** | **81.2** |
| DeiT-III-L | **304** | 75.6 | 79.3 |
| **RDNet-L** | 186 | **81.5** | **83.7** |

**RDNet-T（24M）が iNaturalist で DeiT-III-L（304M）を上回る**——13 倍のパラメータ差を覆す。長尾分類で連結型が強いのは、**特徴の再利用が少数クラスの学習を助ける**という DenseNet 本来の主張と符合する。

### 頑健性 — 精度で負けても OOD で勝つ

| Model | IN | **Avg Shift** | Sketch | R |
|---|---|---|---|---|
| NAT-T | **83.2** | 44.0 | 31.9 | 44.9 |
| **RDNet-T** | 82.8 | **44.7** | **37.0** | **49.0** |
| ConvNeXt-L | 84.3 | 49.9 | 40.1 | 53.5 |
| **RDNet-L** | **84.8** | **52.2** | **44.5** | **56.5** |

論文が明示する通り「**RDNet が競合モデルより ImageNet-1K で低い精度を示す場合でさえ、高い OOD スコアを達成する**」。

### その他の知見

- **解像度ロバスト性**: 画像サイズ 1000 で RDNet-T は約 68% を保つ（DeiT-S 約 34%、DenseNet161 約 44%）。しかも DenseNet161 と違いレイテンシ/メモリが破綻しない
- **CKA**: RDNet は各層で異なる特徴を学ぶ（ConvNeXt は中盤の層同士が似る）
- **Stochastic Depth が効く**: 0 → 0.15 で **+1.2%p**。DenseNet はもともと使っていなかった
- **生成モデルでも有効**（Appendix 0.G）: WGAN の生成器を連結型に置き換えて FID 27.79 → **25.37**

## 弱点

- **-large までしかスケールしていない**（論文が明記。資源の制約）
- **最新モデル（BiFormer / SMT / MogaNet）には精度で負ける**（RDNet-T 82.8 vs BiFormer-S 83.8）
- **メモリで ResNet 系に負ける**: RDNet-T 4.1GB vs ConvNeXt-T 2.7GB / Swin-T 2.6GB。**「メモリ効率を高める」という要旨の主張は DenseNet 比では正しいが ResNet 系比では成立していない**
- **速度優位がバッチサイズ依存**（b1 で 7 倍、b128 で並ぶ）
- **パイロット研究の差が小さい**（平均約 1pt、標準偏差約 2）。しかも **Appendix 0.I.1 で「近代的な訓練レシピが差を縮める」**と自ら記録しており、$\mathcal{E}$ 空間の Sto. Depth / RandErase では差が消えている
- ConvNeXt ほど**網羅的なハイパーパラメータ探索をしていない**（Appendix 0.A.2 で「はるかに軽くスイープする」と明記）

## 系譜・位置づけ

- **[[entities/convnext]] と対をなす**。同じ手続き（現代レシピでベースラインを引き直し、要素を 1 つずつ変えて段階表を作る）を DenseNet に適用。**しかも ConvNeXt のブロック設計をそのまま借りている**（LN・depthwise・7×7・活性化を減らす）
- **[[concepts/skip-connection]] の「加算 vs 連結」という軸に一次情報を与えた**。ViT も Swin も ConvNeXt も内部は加算なので、本論文の主張が正しければ **Transformer 側にも連結を試す余地がある**
- **[[entities/nfnet]] と同型の態度**。「定石とされた要素（NFNet は正規化、RDNet は加算ショートカットと Stochastic Depth 不使用）を疑って再検証する」
- **[[entities/hrnet]] とは別系統だが問題意識が近い**——どちらも「特徴を捨てずに保持する」設計で密予測に強い
- **[[entities/inceptionnext]] と同じ診断・別の処方**。同じ 2024 年、同じく「ConvNeXt の 7×7 depthwise が実機で遅い」から出発するが、InceptionNeXt は**カーネルを 4 分岐に分解**（3×3 + 1×11 + 11×1 + 恒等）することで応じた。本論文の表 4 に **InceptionNeXt-T が 132ms と表中最速のレイテンシ**で載っており、**RDNet-T は 82.8 / 175ms** なので**精度で RDNet、速度で InceptionNeXt**という住み分けになる。両者が変えた軸（ショートカットの型 vs トークン混合器）は直交しており、組み合わせは未検証 ⚠

## 公開

- コード / 重み: <https://github.com/naver-ai/rdnet>
- `timm` に収録（`rdnet_tiny` / `rdnet_small` / `rdnet_base` / `rdnet_large`）
- ImageNet-1K でゼロから訓練（300 エポック、AdamW、**EMA なし**）。訓練設定は Swin / ConvNeXt に準拠

## 関連ページ

- [[sources/rdnet]] — 論文要約（最重要）
- [[translations/rdnet]] — 全文和訳（Appendix 0.A〜0.I 込み）
- [[concepts/skip-connection]] — 加算 vs 連結という軸の整理
- [[entities/convnext]] / [[sources/convnext]] — 対をなす論文。ブロック設計の出どころ
- [[concepts/convolutional-neural-network]] — CNN の系譜
- [[entities/swin-transformer]] / [[concepts/vision-transformer]] — 加算ショートカットを共有する相手
- [[entities/hrnet]] — 「特徴を保持する」別系統の設計
- [[entities/nfnet]] — 「定石を疑う」同型の態度
- [[entities/inceptionnext]] / [[sources/inceptionnext]] — 同じ診断（7×7 depthwise が遅い）に別の処方を出した同年の論文
