---
type: entity
entity_kind: model
aliases: [iBOT, image BERT pre-Training with Online Tokenizer]
tags: [self-supervised, vision-transformer, masked-image-modeling, distillation, bytedance]
related: [[concepts/self-supervised-learning]], [[concepts/masked-image-modeling]], [[concepts/online-tokenizer]], [[concepts/vision-transformer]], [[concepts/knowledge-distillation]]
sources: [[sources/ibot]], [[sources/dinov2-learning-robust-visual-features-without-supervision]]
updated: 2026-05-25
---

# iBOT（手法・モデル）

## 概要

**iBOT** = **image BERT pre-Training with Online Tokenizer**。Zhou Jinghao ら（ByteDance, Johns Hopkins, Shanghai Jiao Tong, UC Santa Cruz）が 2021 年 11 月に発表した自己教師あり手法。**DINO（[[entities/dino]]）の自己蒸留と MIM を、新規の「online tokenizer」（[[concepts/online-tokenizer]]）という発想で統合**したハイブリッド。

- 論文: "iBOT: Image BERT Pre-Training with Online Tokenizer"
- arXiv: 2111.07832（2021 年 11 月）→ ICLR 2022
- コード: <https://github.com/bytedance/ibot>
- 詳細解説: [[sources/ibot]] / 翻訳: [[translations/ibot]]

DINOv2（[[entities/dinov2]]）と DINOv3（[[entities/dinov3]]）の**学習目的関数の直接の源**で、現代 CV SSL の中核的な参照点となる重要論文。

---

## なぜ iBOT が重要か

iBOT 登場時の CV SSL は**2 つの目的関数の片方しか選べない状況**だった：

| | 画像レベル目的（DINO, BYOL, MoCo v3） | パッチレベル MIM（MAE, BEiT） |
|---|---|---|
| 強み | 凍結特徴量、k-NN、画像分類 | dense prediction、fine-tune |
| 弱み | パッチレベル表現がやや弱い | 凍結特徴量が弱い、k-NN が弱い |

**iBOT はこれを単一の枠組み（self-distillation）で同時にやる**ことで両方の長所を取った。これが DINOv2/v3 が「凍結特徴量が線形分類でも密予測でも強い」という驚異的バランスを実現する基礎になった。

加えて、**BEiT のような事前訓練済み dVAE トークナイザを不要にし、訓練を単一段階に簡略化**した。これも DINOv2/v3 のスケーラビリティの礎。

---

## アーキテクチャと学習目的関数

iBOT は **DINO + パッチ単位の self-distillation MIM** という構成。

### 損失の 2 つの項

**(1) [CLS] レベル損失（DINO 由来）**:

$$
\mathcal{L}_{\texttt{[CLS]}} = -P_{\theta'}^{\texttt{[CLS]}}(v)^\top \log P_\theta^{\texttt{[CLS]}}(\hat{u})
$$

- 同じ画像の 2 つの crop（拡張で生成された $u, v$）を student と teacher に通す
- student にはマスク済み、teacher にはマスクなしを入力
- [CLS] トークンの出力分布をクロスエントロピーで一致させる
- 対称化（$\hat{v}, u$ の組も加算）

**(2) MIM 損失（iBOT の新規性、online tokenizer）**:

$$
\mathcal{L}_{\mathrm{MIM}} = -\sum_{i=1}^N m_i \cdot P_{\theta'}^{\mathrm{patch}}(u_i)^\top \log P_\theta^{\mathrm{patch}}(\hat{u}_i)
$$

- マスクされた位置 $m_i = 1$ のみで計算
- student のマスク位置の出力を、teacher の対応する**非マスク**パッチ出力に近づける
- **teacher（EMA 更新される student のコピー）が「online tokenizer」として機能**

**総損失**: $\mathcal{L}_{\texttt{[CLS]}} + \mathcal{L}_{\mathrm{MIM}}$（スケーリングなしの単純合計が最良）

### "Online Tokenizer" とは

iBOT の中心的概念。詳細: [[concepts/online-tokenizer]]

- **BEiT**: 事前訓練された dVAE で離散コードを予測（オフライン tokenizer、多段階訓練）
- **MAE**: 画素を直接予測（tokenizer 不要だが画素は意味的に粗い）
- **iBOT**: **teacher 自身**を tokenizer として動的に使う（オンライン = 学習中に EMA で進化）

これにより：
- 事前 tokenizer 学習が**不要**（単一段階）
- ターゲットが**意味的**（画素より高次）
- DINO と同じ自己蒸留枠組みに**統一**できる
- **ドメイン汎用**（dVAE のドメイン固定問題がない）

### 元の iBOT の主要設計

| 項目 | 値 | 補足 |
|---|---|---|
| Backbone | ViT-S/B/L, Swin-T/{7,14} | パッチサイズ 16 |
| マスキング戦略 | **ブロック単位**（BEiT 由来）| MAE のランダム 75% とは違う |
| 予測比率 $r$ | **ランダム**: 0.5 確率で 0、0.5 確率で [0.1, 0.5] | 訓練の半分は DINO として動作 |
| teacher | student の EMA（DINO と同じ）| momentum 0.996〜1 のコサインスケジュール |
| 損失 | CE for both [CLS] and patch | 単純合計 |
| projection head | **[CLS] とパッチで共有** | DINOv2 では分離に変更 |
| 出力次元 K | 8192 | DINO と同じ規模 |
| トークン化 | **ソフトラベル**（連続分布） | hardmax にすると大幅低下 |
| 事前学習データ | ImageNet-1K（800/400/250 ep）+ ImageNet-22K（80/50 ep） | |
| 最適化 | AdamW, batch 1024, lr 5e-4 × bs/256 | |
| マルチクロップ | 2 global (224²) + 10 local (96²) | DINO と同じ |

---

## 主要結果

### ImageNet-1K Linear Probing

| Method | Arch | Param | Linear |
|---|---|---|---|
| DINO | ViT-S/16 | 21M | 77.0 |
| DINO | ViT-B/16 | 85M | 78.2 |
| **iBOT** | ViT-S/16 | 21M | 77.9 (+0.9) |
| **iBOT** | ViT-B/16 | 85M | **79.5 (+1.3)** |
| **iBOT** | ViT-L/16 | 307M | **81.0** |
| **iBOT (IN-22K)** | ViT-L/16 | 307M | **82.3** |

**より大きなモデルほど DINO に対する利得が大きい** = scalability の良さを示す。

### ImageNet-1K Fine-tuning

| Method | ViT-S/16 | ViT-B/16 | ViT-L/16 | ViT-L/16 (22K, 512²) |
|---|---|---|---|---|
| MAE | - | 83.6 | 85.9 | - |
| BEiT | - | 83.4 | 85.2 | - |
| DINO | 82.0 | 83.6 | - | - |
| **iBOT** | **82.3** | **84.0** | 84.8 | **87.8** |

ImageNet-22K + 512² で **87.8%**（MAE ViT-H と同等）。

### COCO 検出 + ADE20K セグメンテーション（ViT-B/16）

| Method | COCO AP^b | COCO AP^m | ADE20K Lin. mIoU | ADE20K UperNet mIoU |
|---|---|---|---|---|
| Sup. | 49.8 | 43.2 | 35.4 | 46.6 |
| BEiT | 50.1 | 43.5 | **27.4** ↓ | 45.8 |
| DINO | 50.1 | 43.4 | 34.5 | 46.8 |
| **iBOT** | **51.2** | **44.2** | **38.3** | **50.0** |

**BEiT の Lin. mIoU が壊滅的に低い**: dVAE トークナイザが低レベル意味性しか持たないため、線形ヘッドだけでは pixel-level 分類が解けない。iBOT は教師あり + 2.9 mIoU の改善で「**パッチ表現に局所意味性がある**」ことを示す。

### 創発する部位レベル意味性（§4.3）

- パッチトークンの射影出力をクラスタリングすると、**「犬の耳」「車のヘッドライト」「縞テクスチャ」など部位レベルパターンが教師なしで現れる**
- BEiT の dVAE では現れない iBOT 固有の性質
- 後の DINOv3 の PCA 可視化や VGGT の 3D 対応のルーツ

### Robustness（ViT-S/16）

| | DINO | iBOT |
|---|---|---|
| 背景変更 (IN-9) | 96.4 | **96.8** |
| 遮蔽 (S.5) | 64.7 | **65.9** |
| 自然敵対例 (IN-A) | 12.3 | **13.8** |
| 破損 (IN-C ↓) | 51.7 | **48.1** |

MIM 由来の「マスクから推論」訓練が robustness につながる。

---

## 系譜と派生

### 由来（2 つの祖先）
- **[[entities/dino]]**（2021/04, Caron et al.）: [CLS] 自己蒸留損失の出所
- **[[entities/mae]]**（2021/11, He et al.）: 画素 MIM の同時期作（iBOT はやや先か並行）
- **BEiT**（2021/06, Bao et al.）: dVAE トークナイザによる MIM の先駆者、iBOT が乗り越えた対象

### 後続

- **[[entities/dinov2]]**（2023, Oquab et al.）: iBOT の学習目的関数を**ほぼそのまま継承**しつつ、LVD-142M データセット、ViT-g（1.1B）、KoLeo regularizer、SK centering、SwiGLU、Sequence packing、ヘッド分離等の改良を加える
- **[[entities/dinov3]]**（2025, Siméoni et al.）: DINOv2 をさらに ViT-7B × LVD-1689M に scale、**Gram anchoring**（[[concepts/gram-anchoring]]）で long-training dense feature 劣化を解決、RoPE 採用
- **MaskFeat**（Wei et al., 2022）: 似た発想で HOG 特徴量を予測
- **iBOT 自体は ICLR 2022 で発表**（arXiv は 2021 年 11 月）

---

## DINOv2 との詳細な違い

[[sources/dinov2-learning-robust-visual-features-without-supervision]] §4 表 1 が iBOT → DINOv2 の改善を細かく示している：

| 改善 | INet-1k k-NN |
|---|---|
| ベースライン iBOT | 72.9 |
| 再現（DINOv2 著者ら） | 74.5 |
| + LayerScale + Stochastic Depth | 75.4 |
| + 128k prototypes（DINO ヘッド出力次元拡大） | 76.6 |
| + KoLeo 正則化 | 78.9 |
| + SwiGLU FFN | 78.7 |
| + Patch size 14（16→14） | 78.9 |
| + Teacher momentum 0.994（更新を遅く） | 79.4 |
| + warmup スケジュール調整 | 80.5 |
| + Batch size 3k | 81.7 |
| + Sinkhorn-Knopp centering | 81.7 |
| **+ ヘッド分離 = DINOv2** | **82.0** |

つまり **DINOv2 = iBOT + 10 個の実装テクニック** という関係。iBOT の損失関数自体は維持されている。

---

## 限界

[[sources/ibot]] の限界セクションを要約：

- マルチクロップ依存（マルチクロップなしだと ViT-S linear 77.9 → 76.2）
- メモリ +25%、訓練時間 +7.4%（DINO 比）
- マルチクロップ + MIM の安定性問題（Appendix B 全体がこの対策）
- ViT-L 以上での dense feature 劣化（後の DINOv3 が暴いた問題）
- MAE と比較すると IN-1K のみ事前学習時のファインチューニング性能でやや劣る（ViT-L で MAE 85.9 vs iBOT 84.8）

---

## 公開モデル

GitHub <https://github.com/bytedance/ibot> で公開：

| Model | Param | 事前学習データ | Linear | Fine-tune |
|---|---|---|---|---|
| iBOT ViT-S/16 | 21M | IN-1K, 800 ep | 77.9 | 82.3 |
| iBOT ViT-B/16 | 85M | IN-1K, 400 ep | 79.5 | 84.0 |
| iBOT ViT-L/16 | 307M | IN-1K, 250 ep | 81.0 | 84.8 |
| iBOT ViT-B/16 | 85M | IN-22K, 80 ep | 79.0 | 84.4 |
| iBOT ViT-L/16 | 307M | IN-22K, 50 ep | 82.3 | 86.6 / 87.8 (512²) |

---

## なぜ重要か（まとめ）

1. **MIM × Self-Distillation のハイブリッドという新パラダイムの確立**: 後の DINOv2/v3 がすべて継承
2. **Online tokenizer という概念的革新**: BEiT の多段階パイプラインを単一段階に簡略化、ドメイン汎用化
3. **凍結特徴量と密予測の両立**: 当時の SSL の根本的トレードオフを解決した最初の本格事例
4. **部位レベル意味性の創発**: パッチトークンが教師なしで「犬の耳」などのパターンを学習することを実証
5. **ByteDance + JHU の貢献**: Meta（FAIR）一強だった CV SSL に多様性をもたらした

---

## 関連ページ

- [[sources/ibot]]: 原論文の詳細解説
- [[translations/ibot]]: 原論文全訳（Appendix 込み）
- [[concepts/online-tokenizer]]: iBOT の核心アイデアの解説
- 系譜: [[entities/dino]] + [[entities/mae]] → **iBOT** → [[entities/dinov2]] → [[entities/dinov3]]
- 概念: [[concepts/masked-image-modeling]] / [[concepts/self-supervised-learning]] / [[concepts/knowledge-distillation]] / [[concepts/denoising-autoencoder]]
- データ: [[entities/imagenet]]
