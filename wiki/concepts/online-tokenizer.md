---
type: concept
aliases: [Online Tokenizer, オンライントークナイザ]
tags: [ssl, masked-image-modeling, distillation, tokenization]
related: [[masked-image-modeling]], [[knowledge-distillation]], [[self-supervised-learning]]
sources: [[sources/ibot]]
updated: 2026-05-25
---

# Online Tokenizer（オンライントークナイザ）

## 一言で

**「マスク画像モデリング（MIM）において、マスクされたパッチの教師信号を提供する『視覚トークナイザ』として、事前訓練された固定モデルではなく、現在学習中のモデル自身（の EMA teacher）を動的に用いる**」というアイデア。iBOT（[[entities/ibot]], Zhou et al., 2021）が提案。これにより BEiT のような多段階訓練パイプラインが単一段階に簡略化され、トークナイザがドメインデータと同時に進化する。後の DINOv2 / DINOv3 の損失設計の中核にもなった重要概念。

---

## なぜ「トークナイザ」が必要か

MIM（[[concepts/masked-image-modeling]]）は本質的に「マスクされたパッチを残りから予測する」タスクだが、**「何を予測するか」**で大きく性質が変わる：

| MIM 手法 | 予測ターゲット | トークナイザ |
|---|---|---|
| MAE（[[entities/mae]]）| 画素値（MSE） | 恒等写像（pixel = target） |
| BEiT | 離散コードブック ID | **事前訓練済み dVAE**（DALL-E 由来） |
| **iBOT**（[[entities/ibot]]）| teacher の出力分布 | **online tokenizer**（teacher 自身） |

「トークナイザ」という言葉は NLP の WordPiece（BERT のサブワードトークナイザ）から来ている。NLP では「文をどう離散化するか」が言語モデルの基礎で、視覚 MIM でも同様に「画像パッチをどんな教師信号に変換するか」が決定的に重要、という発想。

---

## 既存手法の問題

### 画素ターゲット（MAE 系）

- 「ピクセル」は意味的に粗い
- 結果として **linear probing と k-NN が弱い**（凍結特徴量で意味分類器が作りにくい）

### 事前訓練 dVAE（BEiT）

- **多段階訓練が必要**: dVAE を別途学習する必要がある
- **追加データが必要**: BEiT の dVAE は 2.5 億画像（DALL-E のもの）で訓練
- **ドメイン固定**: dVAE が学習されたドメインに縛られる
- **低レベル意味性しか捕捉しない**（iBOT 論文 §4.3.1 で実証）

### iBOT の発見

iBOT 論文 §3 の鋭い観察：

> BEiT の MIM 損失（Eq. 1）と DINO の自己蒸留損失（Eq. 2）は**同じ形式**をしている。

両者ともに「**ある分布を別の分布にクロスエントロピーで近づける**」構造。BEiT の事前訓練済み tokenizer の代わりに、DINO 流の **EMA で動的に更新される teacher** を入れれば、両者が統合できる。

---

## Online Tokenizer の仕組み

```
                                  ┌──→ teacher network g_t (EMA)
画像 x                              │       └─→ パッチ出力 P_t^patch(x_i)  ←── 教師信号
   ↓                                │
拡張 → ビュー u, v                  │
   ↓                                │
ブロック単位マスク                  │
   ↓ ↓                              │
masked u, masked v                  │
   ↓ ↓                              │
student network g_s                 │
   ↓                                │
student のマスク位置の出力         │
P_s^patch(û_i)  ←─── 損失 ───┐    │
                              │    │
                              ↓    │
                     CE(P_t, P_s) ─┘
                              │
                              ↓
              ← stop-gradient ── teacher へ流さない
                              │
                              ↓
              EMA 更新で teacher → student の重みを継承
```

### 数式

$$
\mathcal{L}_{\mathrm{MIM}} = -\sum_{i=1}^{N} m_i \cdot P_{\theta'}^{\mathrm{patch}}(x_i)^\top \log P_\theta^{\mathrm{patch}}(\hat{x}_i)
$$

- $\theta$: student、$\theta'$: teacher（EMA 更新）
- $m_i$: マスク指示子（マスクされた位置のみ損失を計算）
- $x_i$: teacher への入力（**マスクされていない**パッチ）
- $\hat{x}_i$: student への入力（**マスクされた**パッチに対応する位置）
- $P^{\mathrm{patch}}$: 射影ヘッド + softmax

### 重要な設計判断

1. **teacher と student で射影ヘッドを共有**（$h^{\texttt{[CLS]}} = h^{\mathrm{patch}}$）
   - [CLS] トークンの自己蒸留で獲得された意味性がパッチトークンの MIM 訓練にも転移
   - iBOT のアブレーション（表 17）で shared > non-shared > semi-shared
2. **ソフトラベル（連続分布）を使用**、ハードラベル（one-hot）ではない
   - 画像パッチは意味的に曖昧で、確定的な分類は sub-optimal
   - iBOT アブレーションで hardmax → softmax で大幅改善
3. **centering + sharpening が必要**（DINO 由来）
   - 崩壊回避
4. **teacher は EMA で更新、stop-gradient で勾配は流さない**
   - 標準的な自己蒸留の挙動

---

## なぜ「online」と呼ぶか

| | offline tokenizer（BEiT） | **online tokenizer**（iBOT） |
|---|---|---|
| 訓練タイミング | **事前**に別ジョブで学習 | **同時**に学習（main 訓練と並行） |
| 更新 | 訓練中は固定 | EMA で**動的に**更新され続ける |
| データ | 別データセットでも OK | **同じデータ**で進化 |
| パイプライン | 多段階 | **単一段階** |

「online」= 「学習中に動的に更新される」というニュアンス。online learning の一般用法と同じ。

---

## 後続研究での採用

iBOT の online tokenizer は、その後の SSL 研究で**標準的な選択肢**となった：

### DINOv2（2023, [[entities/dinov2]]）

iBOT の損失をそのまま継承（厳密にはヘッドを分離するなど微調整あり）：

```
L_pre = L_DINO + L_iBOT + 0.1 * L_KoLeo
```

L_iBOT が iBOT の online tokenizer MIM 損失そのもの。

### DINOv3（2025, [[entities/dinov3]]）

iBOT 損失を基盤として継承しつつ、新規の **Gram anchoring**（[[concepts/gram-anchoring]]）を追加。
それでも online tokenizer の発想は維持。

### DINOv3 以降

「**teacher 自身を tokenizer/anchor として使う**」という発想は、Gram anchoring の Gram teacher にも拡張されており、iBOT の online tokenizer 概念の系譜は CV SSL の主流になっている。

---

## 他手法との関係

### vs Knowledge Distillation（[[concepts/knowledge-distillation]]）

通常の KD は「事前訓練済みの大規模 teacher → 小型 student」へ知識を蒸留する。online tokenizer は **「同サイズの student の EMA を teacher として使う self-distillation」** の特殊形。

### vs Mean Teacher（半教師あり学習）

Mean Teacher（Tarvainen & Valpola, 2017）が EMA 更新の teacher を最初に提案。DINO がこれを SSL に持ち込み、iBOT がさらに MIM の教師信号として使った。

### vs Predictor（BYOL）

BYOL は student の最後に追加の MLP predictor を入れて崩壊を防ぐ。iBOT は predictor を使わず、代わりに teacher 出力に対する centering + sharpening で崩壊を防ぐ（DINO と同じ）。

---

## なぜ重要か

1. **MIM の多段階パイプラインを単一段階に簡略化**: dVAE などの事前学習を不要に
2. **意味的に豊かなトークナイザの動的進化**: 訓練が進むほど tokenizer 自体が良くなる「自己改善ループ」
3. **ドメイン汎用性**: 別ドメイン（医療画像、衛星画像など）でも事前訓練 tokenizer を必要としない
4. **DINOv2 / DINOv3 の中核損失**: 現代 CV SSL の標準コンポーネント

---

## 関連ページ

- [[sources/ibot]] — online tokenizer を提案した論文
- [[entities/ibot]] — online tokenizer を採用したモデル
- [[concepts/masked-image-modeling]] — MIM の文脈
- [[concepts/knowledge-distillation]] — online tokenizer は自己蒸留の特殊形
- [[concepts/self-supervised-learning]] — SSL 全般
- [[entities/dinov2]] / [[entities/dinov3]] — online tokenizer を継承する後継モデル
