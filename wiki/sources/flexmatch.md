---
type: source
source_path: raw/papers/FlexMatch_ Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling.md
source_kind: paper
title: "FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling"
authors: [Bowen Zhang, Yidong Wang, Wenxin Hou, Hao Wu, Jindong Wang, Manabu Okumura, Takahiro Shinozaki]
year: 2021
venue: NeurIPS 2021
ingested: 2026-05-27
tags: [semi-supervised-learning, pseudo-label, dynamic-threshold, curriculum-learning, fixmatch, cifar, svhn, stl-10, imagenet, titech, microsoft-research-asia]
translation: "[[translations/flexmatch]]"
---

# FlexMatch: Curriculum Pseudo Labeling によるFixMatch改良

> 原典: [[translations/flexmatch]] ・ `raw/papers/FlexMatch_ Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling.md`
> 著者: Bowen Zhang, Yidong Wang ら（東京工業大学・Microsoft Research Asia）
> 年・会議: 2021 / NeurIPS 2021

## 一言まとめ

FixMatch の「全クラス固定閾値」という設計を「**クラス別動的閾値**（CPL: Curriculum Pseudo Labeling）」に置き換えることで、追加コストなしに CIFAR-10（40 ラベル）で 4.97%（FixMatch: 7.47%）を達成した。特にラベルが極端に少ない場合・クラス数が多い場合・タスクが困難な場合に大きな効果を発揮し、収束速度は FixMatch の 1/5 程度。

## 背景と問題意識

[[entities/fixmatch]] は「τ=0.95 というハードな閾値を全クラスに均等に適用する」設計で MixMatch を大幅に上回った。しかしこの設計には見落としがある：

**クラスによって学習難易度は違う。** 例えば CIFAR-10 の「airplane（飛行機）」と「dog（犬）」では、同じモデルが同じ訓練段階において全く異なる確信度を示すことが多い。固定閾値は「簡単なクラスには高すぎず、難しいクラスには低すぎず」というちょうど良い設定がないジレンマを抱える。

加えて、**訓練初期は閾値を超えるサンプルが少なく、ラベルなしデータを大量に無駄にしている**。FixMatch では訓練中盤まで使われる疑似ラベルが極端に少ない「冷えた」スタートが生じる。

> **直感的な理解**: 「高校の先生が全生徒に同じテストをするのではなく、各生徒の理解度に合わせてテストの難易度を変える」のがカリキュラム学習の発想。FixMatch の固定閾値は「全員に東大入試レベルの問題」を課し続けるようなもので、苦手な学生（困難なクラス）はほとんど勉強できない。

## 提案手法：CPL（Curriculum Pseudo Labeling）

### 学習効果の推定：σ_t(c)

CPLの鍵は「**評価セットやバリデーションセットを使わずにクラスごとの学習状態を推定する**」こと。

方法は単純だ。FixMatch が通常の一貫性損失を計算するたびに、「**閾値 τ を超えてクラス c に予測されたラベルなし例の数**」を数えて記録するだけ：

$$\sigma_t(c) = \sum_{n=1}^{N} \mathbf{1}[\max(p_m(y|u_n)) > \tau] \cdot \mathbf{1}[\arg\max(p_m(y|u_n)) = c]$$

これは損失計算時のボーナス操作なので**追加の推論コストゼロ**。「クラス c を高信頼で予測できるサンプルが多い = クラス c がよく学習されている」という仮定がある。

### 動的閾値の計算

$\sigma_t(c)$ を正規化して $\beta_t(c) \in [0, 1]$ を作り、クラス c の閾値を決める：

$$\beta_t(c) = \frac{\sigma_t(c)}{\max_c \sigma_t}, \quad \mathcal{T}_t(c) = \mathcal{M}(\beta_t(c)) \cdot \tau$$

- **最もよく学習されたクラス**: $\beta_t(c) = 1$ → 閾値 = τ（FixMatch と同じ）
- **学習困難なクラス**: $\beta_t(c)$ が小さい → 閾値 が低い → 疑似ラベルを使いやすくなる

$\mathcal{M}(\cdot)$ はマッピング関数で、デフォルトは凸型の $\mathcal{M}(x) = x/(2-x)$。これにより「学習初期のふらつきを安定させ、学習が進んだ段階で選別基準を急激に上げる」という挙動になる。

### ウォームアップ機構

訓練開始直後はモデルが初期化の偏りで特定クラスに集中して予測しがち（確証バイアス）。σ_t(c) が信頼できない段階では、「まだ疑似ラベルが振られていないサンプルの数」を分母に加えることで全クラスの閾値をゼロからゆっくり上昇させる：

$$\beta_t(c) = \frac{\sigma_t(c)}{\max\left\{\max_c \sigma_t,\; N - \sum_c \sigma_t\right\}}$$

「未使用サンプル数が優勢な間はウォームアップ、使用済みが優勢になったら通常計算に切り替え」という自動判定。これにより訓練初期に多くのラベルなしデータを活用できる「学習ブーム」が生まれる。

### FixMatch との関係

FlexMatch は FixMatch の「最小変更版」として設計されている：

| 設計要素 | FixMatch | FlexMatch (CPL 追加) |
|---|---|---|
| 閾値 | 全クラス固定 τ=0.95 | クラス別動的 T_t(c) |
| 閾値の計算 | なし（定数） | σ_t(c) のオンライン追跡 |
| ウォームアップ | なし | 自動ウォームアップ |
| 弱→強拡張 | そのまま使用 | 変更なし |
| 追加パラメータ | — | **なし** |
| 追加推論コスト | — | **なし** |

CPL は他の SSL 手法（UDA, Pseudo-Labeling）にも同様に適用でき、それぞれ改善をもたらす（Flex-UDA, Flex-PL）。

## 実験結果

### 主要ベンチマーク（最良エラー率 %、WRN 系統）

| データセット | ラベル数 | FixMatch | FlexMatch | 相対改善 |
|---|---|---|---|---|
| CIFAR-10 | 40 | 7.47 | **4.97** | **−33%** |
| CIFAR-10 | 250 | 4.86 | 4.98 | ≒同等 |
| CIFAR-10 | 4000 | 4.21 | **4.19** | 微改善 |
| CIFAR-100 | 400 | 46.42 | **39.94** | **−14%** |
| CIFAR-100 | 2500 | 28.03 | **26.49** | −5.5% |
| STL-10 | 40 | 35.97 | **29.15** | **−19%** |
| STL-10 | 250 | 9.81 | **8.23** | −16% |
| STL-10 | 1000 | 6.25 | **5.77** | −7.7% |
| **SVHN** | **40** | **3.81** | 8.19 | **+115%（悪化）** |
| SVHN | 1000 | 1.96 | 6.72 | 悪化 |
| ImageNet (100K) | — | 43.66 | **41.85** | top-1 |

**収束速度**: CIFAR-100（400 ラベル）で 200K 反復時点の精度が FixMatch 56.35%、FlexMatch **94.29%**。FlexMatch は FixMatch 最終結果に達するのに訓練時間 1/5 未満。

### 重要パターン

- **ラベルが少ないほど差が大きい**: CIFAR-10 は 40 ラベルで −33%、4000 ラベルでは微差
- **クラス数が多いほど差が大きい**: CIFAR-100（100 クラス）>> CIFAR-10（10 クラス）
- **難タスクほど差が大きい**: STL-10（分布外ラベルなしデータ）で特に顕著
- **SVHN では逆転する**: クラス不均衡データセットで CPL が裏目に出る（後述）

## SVHN 失敗事例：クラス不均衡の罠

SVHN（番地の数字認識）では FlexMatch が FixMatch より大幅に悪化する（40 ラベル：8.19% vs 3.81%）。原因を論文は明確に分析している：

SVHN はクラスごとのサンプル数が不均衡（数字の 0 は少ない、1 や 2 は多い）。CPL では $\sigma_t(c)$ が「クラス c に属する疑似ラベル数」なので、サンプルが少ないクラスは「十分に学習できていても」$\sigma_t(c)$ が大きくなれない。結果として低閾値が維持され、ノイズの多い疑似ラベルが学習に使われ続ける。

一方 SVHN は「数字の分類」という比較的簡単なタスクであり、FixMatch の高固定閾値でも問題なく機能する。「簡単かつ不均衡」という組み合わせでは CPL が逆効果になる。

> **設計上の教訓**: CPL の仮定は「ラベルなしデータがクラス間でバランスしている（または近い）」こと。この仮定が崩れる長尾分布やクラス不均衡設定では要注意。論文も「長尾シナリオでの改善が今後の課題」と明示している。

## アブレーション：3 つのコンポーネントの効果

### τ の上限

FlexMatch でも τ=0.95 付近が最適（図 4a）。ただし FlexMatch の τ は「上限」として機能するだけでなく、$\sigma_t(c)$ の計算にも使われるため、FixMatch とは役割が少し異なる。

### マッピング関数

凸型 $x/(2-x)$ > 線形 $x$ > 凹型 $\ln(x+1)/\ln 2$ の順。凸型が優れる理由は「学習初期（$\beta$ が小）は閾値をゆっくり上げてデータをたくさん使い、学習後期（$\beta$ が大）は閾値を急に上げて品質を確保する」という理想的な挙動を示すため。

### ウォームアップ

ウォームアップあり vs なし：CIFAR-10 で +0.2%、CIFAR-100 で +1% の絶対改善。ウォームアップなしの場合、訓練開始直後に閾値が極端に偏り、いくつかのクラスだけ高閾値・他は低閾値になるという不安定な初期状態が生まれる。

### 分布アライメントとの比較

[[sources/fixmatch]] の CIFAR-100 問題（DA 不在でクラス不均衡補正ができない）を解決するアプローチとして、ReMixMatch の Distribution Alignment（DA）を FixMatch に追加する方法と CPL を比較すると、CPL がより効果的（CPL: 39.94% vs DA 追加: 7.16%→論文記載の FixMatch+DA: 40.14%）。ただし両者は異なる問題（クラス多様性 vs クラス分布補正）を解いているため、組み合わせの余地もある。

## 研究上の位置づけ

FlexMatch が CV の SeSL にとって重要なのは、「閾値は固定すべきではない」という設計思想の転換点だからだ。

FixMatch が「質 > 量（τ=0.95 で 98% 捨てても平気）」を証明したのに対し、FlexMatch は「**クラスごとに最適な閾値がある**」ことを示した。これはカリキュラム学習（Bengio et al., 2009）の「簡単な例から難しい例へと段階的に学習する」という考え方の自然な応用だ。

後続の研究への影響：

- **FreeMatch** (2023): クラス別・局所適応の 2 レベル自由閾値。CPL のアイデアを発展させ、クラス偏り問題をさらに改善
- **SoftMatch** (2023): ハード one-hot 閾値をガウス重み付けに変える（高信頼 = 高重み、低信頼 = 低重み、ゼロではない）
- **SemiReward** (2024): 閾値代わりに品質報酬関数で疑似ラベルを選別

また CPL は FixMatch 以外にも適用可能な「プラグイン」として設計されており、Flex-UDA や Flex-PL として他手法の性能も改善した。

> **2021 年以降の潮流**: FlexMatch により「固定閾値」から「適応閾値」へのシフトが確定的になった。2022 年以降の SeSL 論文では「どのように閾値を適応させるか」が主要な設計軸の一つとなっている。

## 用語と略称

- **CPL** = Curriculum Pseudo Labeling（カリキュラム疑似ラベリング）。FlexMatch の核心技術
- **カリキュラム学習（Curriculum Learning）** = 簡単なサンプルから難しいサンプルへと段階的に学習を進める戦略（Bengio et al., ICML 2009）
- **σ_t(c)** = 時間ステップ t におけるクラス c の推定学習効果。閾値を超えてクラス c に予測されたラベルなし例の数
- **β_t(c)** = σ_t(c) を正規化した値（0〜1）。最もよく学習されたクラスが 1.0
- **T_t(c)** = FlexMatch のクラス c 用の動的閾値。M(β_t(c)) × τ で計算
- **凸マッピング関数** = M(x)=x/(2-x)。β が小さいとき閾値をゆっくり上げ、β が大きいとき急に上げる
- **ウォームアップ（threshold warm-up）** = 訓練初期に全クラスの閾値をゼロから徐々に上昇させる機構
- **TorchSSL** = FlexMatch と同時に公開された PyTorch ベースの SSL 統合コードベース（GitHub: TorchSSL/TorchSSL）
- **SeSL** = Semi-Supervised Learning（半教師あり学習）。当 wiki では SSL（自己教師あり学習）との混同を避けるためこの略称を使用
- **WRN / Wide ResNet** = Wide Residual Network。SeSL ベンチマーク標準のバックボーン
- **CIFAR-10/100** = 32×32 画像の 10/100 クラス分類ベンチマーク（Krizhevsky et al.）
- **STL-10** = 96×96 画像、ラベルなしデータが教師あきとは異なる分布という難しい設定

## 関連ページ

- [[translations/flexmatch]]: 日本語全文翻訳
- [[entities/flexmatch]]: スペックシート・実験数値・アルゴリズム詳細
- [[sources/fixmatch]]: FixMatch 論文詳細（FlexMatch の直接の出発点）
- [[entities/fixmatch]]: FixMatch エンティティページ
- [[concepts/semi-supervised-learning]]: 半教師あり学習の全体像（FlexMatch は MixMatch → FixMatch → FlexMatch の系譜）
- [[entities/mixmatch]]: MixMatch（FixMatch の前身）
