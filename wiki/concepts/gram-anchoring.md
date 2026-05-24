---
type: concept
aliases: [Gram Anchoring, Gram 行列正則化, Gram regularization]
tags: [regularization, ssl, training-technique]
related: [[self-supervised-learning]], [[masked-image-modeling]], [[knowledge-distillation]]
sources: [[sources/dinov3]]
updated: 2026-05-24
---

# Gram Anchoring（Gram 行列によるアンカリング）

## 一言で

**「特徴量そのもの」ではなく「パッチ間の Gram 行列（pairwise dot products）」を、過去の自分自身（Gram teacher）に近づける**正則化。DINOv3（[[entities/dinov3]]）で導入され、**ViT-L 以上 × 長時間学習で起きる dense feature 劣化問題**を解決した。スタイル転送由来のテクニックを SSL の文脈で再発見・再応用したもの。

---

## 解決しようとした問題: dense feature の長時間学習劣化

[[entities/dinov2]] や [[entities/ibot]] のような識別型 + MIM ハイブリッド SSL を ViT-L 以上で長時間（200k 反復以上）学習させると、奇妙な現象が起きる：

- **ImageNet classification 精度は単調に上がり続ける** ✓
- **しかし ADE20K segmentation 精度は途中で頭打ちになり、その後**むしろ低下する** ✗
- ViT-7B では segmentation が訓練初期の水準を下回るほど劣化

可視化すると原因がわかる：パッチ間のコサイン類似度マップが「**どこを見ても無関係なパッチに似てしまう**」状態になる。具体的には [CLS] トークンと各パッチトークンのコサイン類似度が訓練中に徐々に上がり、パッチ特徴量が「画像全体」を表すように引き寄せられて**局所性を失う**。

> **補足: なぜこの現象が起きるか（直観）** — global タスク（[CLS] 経由）と dense タスク（パッチ経由）は同じバックボーンを共有している。長時間学習で global タスクの信号がパッチ表現にも漏れ出し、「すべてのパッチが画像全体の意味を持つ」方向に押される。これは画像分類には良いが、「ここは犬の頭、ここは胴体」のような細かい区別が必要な dense タスクには致命的。

---

## Gram Anchoring の発想

**「特徴量の絶対位置は自由にして、相対的な類似度構造だけを保つ」** という発想。

### Gram 行列とは

$P$ 個のパッチがあり、各パッチが $d$ 次元の特徴量を持つとき、Gram 行列は

$$
G = X X^\top \in \mathbb{R}^{P \times P}
$$

で、$G_{ij}$ は「パッチ $i$ と $j$ の内積」、つまりパッチ間の類似度を表す。

特徴量を L2 正規化していれば、$G_{ij}$ はそのまま**パッチ間のコサイン類似度**になる。

> **補足: なぜ Gram 行列か** — 個々の特徴量 $X$ ではなく $XX^\top$ を見ることで、「**特徴量空間がどう回転・反転しても変わらない、パッチ間の相対的な構造**」だけを抽出できる。例えば全パッチの特徴量を 90 度回転させても Gram 行列は変わらない。これにより global タスクで特徴量の絶対値が変動することを許容しつつ、局所構造だけを過去の状態に固定できる。

### 損失関数

$$
\mathcal{L}_{\text{Gram}} = \| X_S X_S^\top - X_G X_G^\top \|_F^2
$$

- $X_S$: student（学習中のネットワーク）の出力パッチ特徴量行列（L2 正規化済み）
- $X_G$: **Gram teacher**（早い段階の teacher のスナップショット、まだ dense feature が綺麗だった頃）のパッチ特徴量行列
- $\| \cdot \|_F$: フロベニウスノルム（行列の各要素を二乗して足して平方根）

学習中、student の Gram 行列を Gram teacher のそれに近づける。

### Gram teacher の選び方

- DINOv3 では訓練の最初の 200k〜1M 反復のスナップショットを Gram teacher として使用
- Gram anchoring は遅く（1M 反復後）適用しても効果がある
- 10k 反復ごとに Gram teacher を主 EMA teacher に更新（"refinement step"）

> **補足: なぜ早期の teacher を使うか** — まさにその時期がまだ dense feature が「綺麗」だった時期だから。Gram teacher は「将来の自分が忘れてしまうであろう dense 情報」を時間カプセルのように保持する役割。

---

## 高解像度 Gram teacher（§4.3）

DINOv3 の追加トリック：

1. Gram teacher には**入力画像の 2 倍解像度**で forward させる（256² → 512²）
2. 出力 feature map を **bicubic で 2× ダウンサンプリング**して student と同じサイズに揃える
3. ダウンサンプリングされた feature map から Gram 行列を計算

これで「高解像度由来の滑らかで一貫性の高いパッチ表現」を低解像度 student に蒸留できる。ADE20K で追加 +2 mIoU の効果。

---

## なぜ効くのか（直観と実験）

### iBOT 損失への正の影響

DINOv3 論文 §4.2 図 7 で観察された興味深い現象：Gram 目的を加えると **iBOT 損失が劇的に速く減少する**。一方 DINO 損失への影響は小さい。

これは Gram と iBOT が「**パッチレベル一貫性**」という同じ方向に影響することを示唆。Gram teacher の安定性が iBOT の最適化を助けている。

### Dense vs Global の分離

Gram anchoring は **global 性能をほぼ犠牲にせずに dense 性能を回復**する：

| | ADE20K (dense) | ImageNet (global) |
|---|---|---|
| 200k 反復時点 | 高い | 高い |
| 1M 反復、Gram なし | **劣化** | さらに改善 |
| 1M 反復、Gram あり | **回復＋向上** | 改善維持 |

これは「**dense と global の信号を構造的に分離する**」というDINOv3 の主張を裏付ける。

---

## スタイル転送との歴史的接続

Gram 行列を使った損失は**新しいアイデアではない**。Gatys ら（2015）の neural style transfer で：

- 画像の「**スタイル**」は特徴量チャネル間の Gram 行列（チャネル × チャネル）で表現
- 「**コンテンツ**」は特徴量自体
- スタイル画像とコンテンツ画像の Gram 行列を組み合わせて新画像を生成

DINOv3 は「**チャネル × チャネル**」ではなく「**パッチ × パッチ**」の Gram 行列を使い、「過去の自分のスタイルを保つ」という新しい使い方をした。

---

## 他手法との関係

### 知識蒸留との違い

通常の知識蒸留（[[concepts/knowledge-distillation]]）は teacher の出力分布や特徴量を直接マッチさせる。Gram anchoring は**特徴量を直接マッチさせない**点が違う。

| | 通常の KD | Gram Anchoring |
|---|---|---|
| 何をマッチさせるか | 特徴量 / ロジット直接 | パッチ間の Gram 行列 |
| 特徴量の自由度 | 制約される | 自由（回転・スケール変換 OK）|
| 用途 | 知識転送・モデル圧縮 | 学習の安定化・dense 構造の保持 |

### PEspatial / AM-RADIO との違い

これら凝集モデルも「**student と teacher のパッチ間コサイン類似度を高く保つ**」損失を使う（§2 Related Work 参照）。違いは：

- **PEspatial / AM-RADIO**: teacher は SAM など**強い教師ありモデル**から固定
- **DINOv3 Gram anchoring**: teacher は **SSL モデル自体の早期スナップショット**から動的に取得

「self-distillation の中で自分自身の過去を teacher に使う」という再帰的・経済的な発想。

---

## 適用範囲と今後の展望

Gram anchoring の発想は SSL 以外にも応用可能：

- **長時間学習で何かが劣化する系**全般（強化学習の policy 安定化、LLM の継続学習など）
- **複数目的が衝突する系**で「どちらかを犠牲にせずに両方保つ」工夫
- **領域固有の構造を保ちたい場合**（医療画像で解剖学的構造を保ったまま意味的学習を進める等）

DINOv3 以降、SSL 研究で標準ツールになる可能性が高い。

---

## 関連ページ

- [[sources/dinov3]] — Gram anchoring を提案した論文の詳細解説
- [[entities/dinov3]] — Gram anchoring を採用したモデルファミリ
- [[concepts/self-supervised-learning]] — Gram anchoring が解決する問題の文脈
- [[concepts/masked-image-modeling]] — iBOT 損失と Gram 目的が相互に強化し合う関係
- [[concepts/knowledge-distillation]] — 蒸留の一形態としての位置づけ
