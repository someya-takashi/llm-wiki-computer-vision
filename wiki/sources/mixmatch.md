---
type: source
source_path: raw/papers/MixMatch_ A Holistic Approach to Semi-Supervised Learning.md
source_kind: paper
title: "MixMatch: A Holistic Approach to Semi-Supervised Learning"
authors: [David Berthelot, Nicholas Carlini, Ian Goodfellow, Avital Oliver, Nicolas Papernot, Colin Raffel]
year: 2019
venue: NeurIPS 2019
ingested: 2026-05-27
tags: [semi-supervised-learning, mixup, consistency-regularization, entropy-minimization, cifar, svhn, stl-10, wide-resnet, differential-privacy, label-guessing, sharpening]
translation: "[[translations/mixmatch]]"
---

# MixMatch: A Holistic Approach to Semi-Supervised Learning

> 原典: [[translations/mixmatch]] ・ `raw/papers/MixMatch_ A Holistic Approach to Semi-Supervised Learning.md`
> 著者・年・会議: Berthelot et al., 2019, NeurIPS 2019
> arXiv: 1905.02249

## 一言まとめ

**半教師あり学習（ラベルあり+なしデータを使う学習）の 3 大アプローチを 1 本のアルゴリズムに統合し、CIFAR-10（250 ラベル）でエラー率を 11.08% まで下げた**（従来最良の VAT が 36.03%）。3 つのアプローチとは「一貫性正則化（同じ画像の 2 つの拡張は同じ予測であるべき）」「エントロピー最小化（ラベルなしデータに対して確信を持った予測をする）」「MixUp 正則化（2 つの例を凸混合して決定境界を平滑化する）」である。

> **用語補足: 半教師あり学習 vs 自己教師あり学習**
> - **半教師あり学習（Semi-Supervised Learning, 略称は SeSL あるいは本文では "SSL" と表記されることが多いが、当 wiki では SSL＝自己教師あり学習）**: **少量のラベルあり**データ＋大量のラベルなしデータを組み合わせる。クラスラベルの教師信号が存在する点が自己教師あり学習と異なる。
> - **自己教師あり学習（Self-Supervised Learning, 当 wiki での SSL）**: ラベルを一切使わず、データ自体の構造（拡張の不変性など）から表現を学習する。SimCLR/BYOL/DINO がこちら。

## 背景と問題意識

2019 年当時、半教師あり学習の代表的な手法はそれぞれ異なる原理で動いていた：

| 手法 | 原理 | 弱点 |
|---|---|---|
| **Π-Model / Mean Teacher** | 一貫性正則化（拡張後も同じ予測を） | ソフトターゲットが高エントロピー（確信なし） |
| **Pseudo-Label** | エントロピー最小化（高信頼のラベルを使う） | ハードラベルの閾値設定が難しい |
| **VAT**（Virtual Adversarial Training） | 最悪方向への摂動に対して一貫性を保つ | 計算コストが高い、ドメイン固有 |
| **MixUp** | ラベルあり例間を凸補間して正則化 | ラベルなしに直接適用しにくい |

これらを組み合わせた研究が存在しなかった。MixMatch はすべてを 1 本のアルゴリズムで統合し、「部分の総和以上」の性能を示した。

## 提案手法：MixMatch の仕組み

<figure>

![](../../raw/assets/mixmatch/fig1.png)

<figcaption>図1: ラベル推定（label guessing）プロセス。ラベルなし画像に K 回データ拡張を適用し、分類器に通す。全予測を平均して温度シャープニングでエントロピーを下げる。</figcaption>
</figure>

MixMatch は入力（ラベルあり $\mathcal{X}$ + ラベルなし $\mathcal{U}$）から処理済みバッチ $\mathcal{X}^{\prime}$, $\mathcal{U}^{\prime}$ を生成する関数として設計される。

### ステップ 1: ラベル推定（Label Guessing）

ラベルなし画像 $u_b$ に $K$ 回データ拡張を適用し、各予測の平均を取る：

$$\bar{q}_{b}=\frac{1}{K}\sum_{k=1}^{K}{\rm p}_{\rm{model}}(y\mid\hat{u}_{b,k};\theta)$$

これは「モデルが安定して同じクラスを予測しているか」を確認する仕組み（一貫性正則化の応用）。

### ステップ 2: シャープニング（Sharpening）

平均予測 $\bar{q}_b$ を温度 $T$ で変換してエントロピーを下げる（エントロピー最小化の実装）：

$$q_b = \operatorname{Sharpen}(\bar{q}_b, T)_i := \frac{p_i^{1/T}}{\sum_j p_j^{1/T}}$$

$T=0.5$ に設定。$T \to 0$ でデルタ分布（one-hot）に収束。温度シャープニングは DINO/DINOv2 の `centering + sharpening` に概念的に受け継がれる。

### ステップ 3: MixUp（ラベルあり・なしを横断した凸混合）

ラベルあき例 $\hat{\mathcal{X}}$ とラベルなし例（推定ラベル付き）$\hat{\mathcal{U}}$ を結合してシャッフルし、各例を「もう一方の例との凸組み合わせ」に置き換える：

$$\lambda \sim \text{Beta}(\alpha, \alpha), \quad \lambda' = \max(\lambda, 1-\lambda)$$
$$x' = \lambda' x_1 + (1-\lambda') x_2, \quad p' = \lambda' p_1 + (1-\lambda') p_2$$

`max` を取る修正（バニラ MixUp にはない）がポイント。これにより $x'$ が必ず $x_1$ 側に寄る → バッチ順序を保持してラベルあき・なしの損失を別計算できる。

### 損失関数

$$\mathcal{L} = \underbrace{\frac{1}{|\mathcal{X}'|}\sum \operatorname{H}(p, {\rm p_{\rm model}})}_{\text{ラベルあき: クロスエントロピー}} + \lambda_{\mathcal{U}} \underbrace{\frac{1}{L|\mathcal{U}'|}\sum \|q - {\rm p_{\rm model}}\|_2^2}_{\text{ラベルなし: Brier スコア（二乗 L2）}}$$

ラベルなし側を二乗 L2（Brier スコア）にする理由：クロスエントロピーは誤った予測に対して unbounded（無限大になりうる）だが、$L_2$ は有界 → ラベルなしの誤ったラベル推定が損失を爆発させない。

### ハイパーパラメータ（ほぼ固定）

| パラメータ | デフォルト値 | 意味 |
|---|---|---|
| $T$ | 0.5 | シャープニング温度 |
| $K$ | 2 | 拡張数 |
| $\alpha$ | 0.75 | MixUp の Beta 分布パラメータ |
| $\lambda_{\mathcal{U}}$ | 75〜250（データセット依存） | ラベルなし損失の重み（訓練初期に 16,000 ステップかけてウォームアップ） |

## 主要な実験結果

### 標準ベンチマーク（Wide ResNet-28, 150 万パラメータ）

**CIFAR-10（表 5 より）**:

| 手法 | 250 ラベル | 4000 ラベル |
|---|---|---|
| Π-Model | 53.02% | 17.41% |
| PseudoLabel | 49.98% | 16.21% |
| Mixup のみ | 47.43% | 13.15% |
| VAT | 36.03% | 11.05% |
| Mean Teacher | 47.32% | 10.36% |
| **MixMatch** | **11.08%** | **6.24%** |
| 完全教師あり（5万枚） | — | 4.17% |

MixMatch は 4,000 ラベルで Mean Teacher の 250 ラベル相当の性能を達成（16 分の 1 のラベルで同等）。

**SVHN（250 ラベル）**: MixMatch 3.78%（Mean Teacher: 6.45%）

**STL-10（1,000 ラベル）**: MixMatch 10.18%（従来 SOTA の 5,000 ラベル版 11.20% を超える）

### 差分プライバシーへの応用

PATE フレームワーク（教師アンサンブルがノイズ付き投票で学生を指導するプライバシー保護フレームワーク）と組み合わせた場合：

| 手法 | 精度 | プライバシー損失 $\varepsilon$ |
|---|---|---|
| VAT（従来 SOTA） | 91.6% | 4.96 |
| **MixMatch + PATE** | **95.21%** | **0.97** |

$e^{\varepsilon}$ でプライバシーを測るため、改善は約 $e^4 \approx 55$ 倍。$\varepsilon < 1$ はかなり強いプライバシー保証。

## アブレーション研究の主な知見

表 4 の CIFAR-10 / 250 ラベルでの結果（エラー率 %）：

| 削除する要素 | 結果 | 重要度 |
|---|---|---|
| MixUp 完全除去 | **39.11%** | 最も重要（28pt 劣化） |
| シャープニング除去（T=1） | 27.83% | 非常に重要（16pt 劣化） |
| 分布平均除去（K=1） | 17.09% | 重要（5pt 劣化） |
| ラベルあきのみ MixUp | 32.16% | ラベルなしへの MixUp 適用が重要 |
| **完全 MixMatch** | **11.08%** | — |

**MixUp なしが最も大きな劣化**。ラベルあり・なしを横断する MixUp が鍵。シャープニングも単独で 16pt 差をつける重要な要素。EMA は逆に MixMatch には不要（SVHN では Mean Teacher の EMA が有効だが、MixMatch に追加してもほぼ変化なしか若干悪化）。

**ICT（Interpolation Consistency Training）との比較**: ICT は「MixUp をラベルなしのみ + シャープニングなし + EMA でラベル推定」という MixMatch の特殊ケースとして解釈できる。250 ラベルで 38.60%（MixMatch は 11.08%）— ラベルあき-なし横断 MixUp とシャープニングの重要性を示す。

## 限界と批判的視点

1. **画像ドメイン特化**: データ拡張（ランダムクロップ + 水平反転）は画像に特有。テキスト・音声への直接適用は困難。
2. **ラベルなし分布の仮定**: STL-10 では明示されているが、ラベルあり・なしの分布が大きくずれる場合は性能が劣化する可能性がある。
3. **閾値なしの生のラベル推定**: Pseudo-Label と異なり信頼度の閾値を設けない。低信頼のラベル推定も全部使う点は設計上の選択だが、ノイズの多い推定が損失に入り込む。
4. **後継手法による改善**: FixMatch（Sohn et al., 2020）が「高信頼度のラベルなし例のみを疑似ラベルとして使用 + RandAugment」でさらに改善（CIFAR-10 / 250 ラベルで 5.07%）。ReMixMatch（同じ著者, 2020）がさらに拡張。
5. **ハイパーパラメータ $\lambda_{\mathcal{U}}$ のデータセット依存性**: CIFAR-10: 75、SVHN: 250、STL-10: 50 と大きく異なる。新しいデータセットへの適用時に調整が必要。

## SimCLR/BYOL との比較（思想的な接続）

MixMatch（半教師あり, 2019）は、後の SimCLR（自己教師あり, 2020）や BYOL（2020）と以下の点で思想的に共通する：

| 要素 | MixMatch | SimCLR / BYOL |
|---|---|---|
| データ拡張の利用 | K 回拡張 → 平均予測（一貫性の確認） | 2 view → 表現を引き寄せる |
| エントロピー最小化 | シャープニング（T=0.5） | DINO の sharpening に受け継がれる |
| ラベルの利用 | 少量のラベルあり + ラベルなし | ラベルなしのみ（pretraining） |
| 目的 | ダウンストリームタスクを直接解く | 汎用表現の事前学習 |

MixMatch の貢献は「ラベルが少ない場合でもラベルを最大限活用する方法論」であり、SimCLR/BYOL の「ラベルなしで汎用表現を作る」とは動機が異なる。現代では両者の組み合わせ（SSL 事前学習 → 少量ラベルでファインチューン）が主流。

## 用語と略称

- **半教師あり学習（Semi-Supervised Learning）**: ラベルあり少量 + ラベルなし大量を組み合わせる学習。本論文での「SSL」。当 wiki の「SSL（自己教師あり学習）」とは別概念（§1 補足参照）
- **MixUp**: Beta 分布でサンプルした混合比で 2 つの例を凸補合する正則化手法（Zhang et al., 2017）
- **Label Guessing**: K 回拡張の予測を平均してラベルなし例の疑似ラベルを生成する操作
- **Sharpening（シャープニング）**: 温度パラメータ $T$ で確率分布を「先鋭化」してエントロピーを下げる操作
- **Consistency Regularization（一貫性正則化）**: 拡張前後で同じ予測を出力するよう促す損失
- **Entropy Minimization（エントロピー最小化）**: ラベルなしデータへの予測を確信のある低エントロピー分布に近づける
- **Brier スコア**: マルチクラス分類の予測確率と正解の二乗 L2 距離。有界で外れ値に頑健
- **Wide ResNet-28（WRN-28）**: 28 層の幅広い ResNet。本論文のデフォルトバックボーン（150 万パラメータ版）
- **CIFAR-10**: 10 クラス、50,000 訓練 + 10,000 テスト画像の小規模画像分類ベンチマーク
- **SVHN**: Street View House Numbers。数字認識の実世界画像ベンチマーク
- **STL-10**: 半教師あり学習専用設計のベンチマーク。5,000 ラベルあき + 100,000 ラベルなし
- **PATE**: Private Aggregation of Teachers' Ensembles。差分プライバシー学習フレームワーク
- **VAT**（Virtual Adversarial Training）: 出力分布を最も大きく変える方向への摂動に対して一貫性を保つ手法
- **Mean Teacher**: 学生と EMA 教師の予測を一致させる半教師あり学習手法（Tarvainen & Valpola, 2017）
- **ICT**（Interpolation Consistency Training）: MixUp をラベルなしのみに適用した MixMatch の特殊ケース
- **差分プライバシー（Differential Privacy）**: 訓練データのどの 1 点を追加/削除してもモデルの出力分布がほぼ変わらないことを保証する数学的フレームワーク

## 関連ページ

- [[concepts/semi-supervised-learning]]: 半教師あり学習全体の系統
- [[entities/mixmatch]]: MixMatch モデルの主要数値まとめ
- [[sources/simclr]]: 一貫性正則化の思想を自己教師あり学習に発展させた手法
- [[sources/byol]]: 非対比 SSL（自己教師あり）の先駆け
- [[sources/dino-emerging-properties-in-self-supervised-vit]]: シャープニングを自己教師あり学習に応用した DINO
- [[concepts/self-supervised-learning]]: 自己教師あり学習（ラベルなし事前学習）—半教師あり学習の比較対象
