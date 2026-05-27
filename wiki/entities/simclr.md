---
type: entity
entity_kind: model
aliases: [SimCLR, "A Simple Framework for Contrastive Learning"]
tags: [contrastive-learning, ssl, representation-learning, resnet, data-augmentation]
related: [[concepts/contrastive-learning]], [[concepts/self-supervised-learning]], [[entities/moco]], [[entities/dino]]
sources: [[sources/simclr]]
updated: 2026-05-27
---

# SimCLR

## 概要

**SimCLR**（A Simple Framework for Contrastive Learning of Visual Representations）は Google Brain の Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton が 2020 年に発表した自己教師あり視覚表現学習フレームワーク。「**特殊なアーキテクチャもメモリバンクも要らず、4 つのシンプルな要素を正しく組み合わせるだけで対比学習は劇的に改善できる**」という主張と体系的なアブレーション実験で、2020 年の SSL の標準ベースライン的立場を確立した。

- 論文: "A Simple Framework for Contrastive Learning of Visual Representations"
- arXiv: 2002.05709（2020）
- 会議: ICML 2020
- 所属: Google Brain（Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton）

---

## 4 つのコンポーネント

| コンポーネント | 詳細 | 設計の核心 |
|---|---|---|
| **データ拡張** | ランダムクロップ＋カラー歪み＋ガウシアンブラーを組み合わせる | 拡張の**構成**（特にクロップ×カラー歪み）が決定的 |
| **エンコーダ f(·)** | ResNet（主に ResNet-50 または大型版） | 下流タスクには **h**（プーリング後）を使う |
| **射影ヘッド g(·)** | 非線形 MLP（h → z, 128次元） | 訓練後は捨てる；h に情報を保持させるバッファ |
| **NT-Xent 損失** | バッチ内 2N 例でコサイン類似度×温度の softmax | $\ell_2$ 正規化＋適切な温度 τ が必須 |

---

## 主要な結果

### ImageNet 線形評価 top-1

| モデル | top-1 | 備考 |
|---|---|---|
| SimCLR (ResNet-50) | 69.3% | 以前の SOTA +5.5% |
| SimCLR (ResNet-50 2×) | 74.2% | |
| SimCLR (ResNet-50 4×) | **76.5%** | 教師あり ResNet-50 と同等 |

### 半教師あり学習（top-5）

| ラベル率 | SimCLR (4×) |
|---|---|
| 1% | 85.8% |
| 10% | 92.6% |

---

## SimCLR が明らかにした知見

1. **拡張の構成 > 個々の拡張**: 複数を組み合わせて初めて有効。特にランダムクロップ×カラー歪みが決定的
2. **非線形射影ヘッドが 10% 以上の改善**: 対比学習で最大の単独効果を持つ設計要素
3. **大バッチほど負例が豊富で収束が速い**: ただし長く訓練すれば小バッチでも追いつく
4. **SSL は大モデルの恩恵が教師ありより大きい**: スケール時の性能ギャップが縮まる
5. **対比学習は教師あり学習より強い拡張を必要とする**: 強い色拡張は教師ありには不要だが SSL には有効

---

## 後継・発展

| 手法 | SimCLR との関係 |
|---|---|
| **MoCo v2 / v3** | SimCLR の設計（射影ヘッド、強拡張）を吸収し momentum encoder と組み合わせ |
| **BYOL** | 負例なし。SimCLR の大バッチ要求を解消 |
| **SimSiam** | BYOL から momentum も除き、stop-gradient だけで崩壊を防ぐ |
| **DINO** | 自己蒸留として再解釈。ViT に特化し強力な密特徴量を実現 |
| **iBOT / DINOv2** | SimCLR 系の contrastive 損失 + MIM 損失のハイブリッド化で dense 性能を大幅向上 |
| **CLIP** | クロスモーダル（画像-テキスト）版対比学習。NT-Xent（InfoNCE）を 4 億スケールに |

---

## 関連ページ

- [[sources/simclr]]: 詳細な論文要約
- [[translations/simclr]]: 日本語全文翻訳
- [[concepts/contrastive-learning]]: SimCLR が代表する対比学習パラダイム
- [[concepts/self-supervised-learning]]: SSL 系統における SimCLR の位置づけ
- [[entities/dino]]: SimCLR から発展した自己蒸留 SSL の代表
- [[entities/clip]]: SimCLR の InfoNCE を画像-テキストに拡張した WSL の代表
