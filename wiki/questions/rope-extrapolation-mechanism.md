---
type: question
asked: 2026-06-10
question: "RoPE が学習データにない相対位置にも適応できるのはなぜでしょうか？例えば学習時の最大系列長が 1000 の場合は、推論時に 1 番目と 2000 番目の相対関係はわからないのではないでしょうか？"
sources_used: ["[[concepts/rotary-position-embeddings]]", "[[sources/roformer]]", "[[sources/yarn]]", "[[entities/qwen2-vl]]", "[[entities/qwen3-vl]]", "[[entities/dinov3]]", "[[questions/vit-dynamic-resolution-evolution]]"]
---

# RoPE は学習範囲を超えた相対位置にどこまで適応できるか — 形式的可能性と実用的破綻

## 結論を先に

**ユーザの直感は半分正しい**:

1. ✅ **形式的には RoPE は任意の相対位置を計算できる**: 回転角 $m\theta_i$ は $m$ の連続関数で、学習可能パラメータを持たないため、$m=2000$ でも数学的には計算可能
2. ⚠️ **しかし実用的には学習範囲を大きく超えると性能が崩壊する**: 訓練時に「**低周波次元での大きな回転角**」を一度も見ていないため、attention の重み行列がその回転角に対する適切な内積パターンを学習していない
3. 🔧 **これを解決するために必要なのが [[sources/yarn|YaRN]] / NTK-aware / Position Interpolation 系の RoPE 拡張手法**

つまり「**数式は計算できる**」と「**モデルが意味のある attention を計算できる**」の **2 段階を区別する**のが鍵。前者は数学、後者は学習器の経験範囲の問題。

---

## なぜ「形式的には動くように見える」のか — 連続関数としての RoPE

[[concepts/rotary-position-embeddings|RoPE]] の核心は、位置 $m$ にあるトークンの query/key ベクトルを **角度 $m\theta_i$ だけ回転**させることです（$\theta_i = b^{-2i/d}$、$b = 10000$、$d$ は次元数）。

3 つの性質：

- **回転角 $m\theta_i$ は $m$ の連続関数**: $m = 2000$ という値でも $\cos(m\theta_i), \sin(m\theta_i)$ は確定値として計算できる
- **学習可能パラメータなし**: 「位置 1, 2, ..., 196 用ベクトル」のような離散辞書を持たない（cf. オリジナル ViT の学習可能 1D 位置埋め込み）
- **内積に現れるのは相対位置 $m-n$ のみ**: $(R_m {\boldsymbol q})^{\intercal} (R_n {\boldsymbol k}) = {\boldsymbol q}^{\intercal} R_{n-m} {\boldsymbol k}$

つまり [[sources/roformer|RoFormer 原論文]] の数式は「**位置インデックス 2000 にも形式的には対応可能**」と言える。これが [[concepts/rotary-position-embeddings]] の「あらゆる解像度で動く」という記述の根拠です。

> [[concepts/rotary-position-embeddings]] §「DINOv3 における RoPE の使い方」より:
> > RoPE は「位置 $m$ に対して角度 $m\theta$ 回転」する操作。$m$ は連続値として扱えるので、訓練時に見たことがない位置（パッチ数）でも、回転角を計算するだけで対応できる。学習可能位置埋め込みは「位置 1, 2, ..., 196 用のベクトル」を訓練したきり拡張できないのと対照的。

**これは「数式的には」正しい**。学習可能 1D 位置埋め込みと比べた相対的優位性は確かに大きい。しかし、これだけでは「実用的にうまく動く」を保証しない。

---

## しかし実際には性能が崩壊する — ユーザの直感が正しい部分

[[sources/yarn|YaRN]] 原論文の図 1 が示す通り：

- **Llama 2（訓練 4k 文脈）**: 8k で perplexity が爆発、10k で完全崩壊
- **Position Interpolation なしの素の RoPE**: 訓練範囲を超えると即座に破綻

これは [[sources/roformer]] §4.5.5 で著者 Su Jianlin 自身が認めた限界（「長文脈外挿の挙動について理論的説明はない」）の正体。

> [[sources/yarn]] §1 より:
> > One reoccurring limitation with positional encodings is the inability to generalize past the context window seen during training. While some methods such as ALiBi are able to do limited generalization, **none are able to generalize to sequences significantly longer than their pre-trained length**.

つまり **「RoPE は数学的には任意位置に対応するが、実際の言語モデルは訓練長を大きく超えると崩壊する」** が業界共通の認識。

---

## なぜ崩壊するのか — 周波数次元ごとの「分布内 vs 分布外」分析

[[sources/yarn]] §3.1-3.2 が解明した核心の答え：**RoPE の各次元 $i$ は異なる波長 $\lambda_i$ を持ち、訓練中に経験した回転範囲が次元ごとに大きく違う**。

### 波長の定義

$$\lambda_i = \frac{2\pi}{\theta_i} = 2\pi b^{2i/d}$$

これは「次元 $i$ で完全な 1 回転（$2\pi$）を行うのに必要なトークン長」。

### 具体例: $d=64, b=10000, L_{\text{train}}=1000$

各次元の波長と訓練中に見た回転数を計算：

| 次元 $i$ | $\theta_i$ | 波長 $\lambda_i$ | 訓練中に見た回転数 $L/\lambda_i$ | 訓練分布 |
|---|---|---|---|---|
| **0（最高周波）** | 1 | 6.28 | **159 回**（完全周回 × 159）| ✅ **完全分布内** |
| 8 | $10000^{-1/4} \approx 0.1$ | 62.8 | 15.9 回 | ✅ 分布内 |
| 16（中周波）| $10000^{-1/2} = 0.01$ | 628 | 1.59 回 | ⚠️ 境界 |
| 24 | $10000^{-3/4} \approx 0.001$ | 6283 | 0.16 回 = **16%** | ❌ 1 回転未満 |
| **31（最低周波）** | $\approx 1/8788$ | **55,000** | **0.018 回 = 1.8%** | ❌ **1 回転すら見ていない** |

### 鍵となる観察

**RoPE では低周波次元（高い $i$）になるほど、訓練時に経験した回転範囲が狭くなる**:
- 高周波次元: 完全な「角度 0〜$2\pi$ → 0〜$2\pi$ → ...」のパターンを何度も経験
- 低周波次元: 「角度 0〜0.113 ラジアン（≒6.5°）」までしか経験しない（$i=31$ で $L=1000$ の場合）

### ユーザの具体例: 相対位置 $m-n = 1999$

ここで相対位置 $m-n = 1999$（学習時上限 1000 の約 2 倍）を計算：

| 次元 $i$ | 回転角 $1999 \cdot \theta_i$ | 訓練分布内か |
|---|---|---|
| 0 | $1999$ rad = **318 周分**（mod $2\pi$ で訓練分布内）| ✅ |
| 8 | $1999 \times 0.1 = 199.9$ rad ≈ 31.8 周 | ✅ |
| 16 | $1999 \times 0.01 = 20$ rad ≈ 3.18 周 | ⚠️ 境界 |
| 24 | $1999 \times 0.001 = 2$ rad ≈ 0.32 周 | ❌ 訓練時は 0〜0.16 周のみ |
| **31** | $1999 \times 0.000114 = 0.228$ rad ≈ **3.6%** | ❌ **訓練時は 0%〜1.8% のみ → 完全に分布外** |

**ここが破綻のメカニズム**:

- 高周波次元 ($i=0$ 等): 角度 $1999\theta_0 \pmod{2\pi}$ は訓練中に何度も経験した値の周期内
- **低周波次元 ($i=31$ 等)**: 「**3.6% という回転角**」は訓練中に **0%〜1.8% しか見ていない範囲の外側**
- Attention の重み行列 $W_q, W_k$ は **この未経験の回転角に対する適切な内積パターンを学習していない**
- Attention の内積はすべての次元の貢献の和なので、**低周波次元が異常値を返すと内積全体が壊れる**

### 直感的な例え

> ピアノの鍵盤を考える。**高音側のキー**は短い周期で繰り返される（毎オクターブで同じ音名が戻る）ので、初心者でも未経験の音域に対応できる。**低音側のキー**は周期が長く、訓練で 1 オクターブの一部しか聞いたことがなければ、次のオクターブの音は予測できない。
>
> RoPE の各次元は異なる「音域」を担当するので、**低音側（低周波次元）が訓練範囲を超えると、全体の attention が壊れる**。

---

## なぜ「数式が成り立つ」だけでは不十分か

RoPE の理論的性質：
$$\langle R_m {\boldsymbol q}, R_n {\boldsymbol k}\rangle = {\boldsymbol q}^{\intercal} R_{n-m} {\boldsymbol k}$$

は **「内積が相対位置 $m-n$ のみに依存する」** ことを保証する。**しかしこの式の値が「意味のある attention 値」になるかは別問題**。

理由：

1. ${\boldsymbol q}, {\boldsymbol k}$ は **学習された** ベクトル（埋め込みと $W_q, W_k$ の積）
2. 学習中、これらのベクトルは **訓練範囲内の $m-n$ に対する $R_{n-m}$ で内積を最適化される**
3. 訓練範囲を超えた $R_{n-m}$ は **${\boldsymbol q}, {\boldsymbol k}$ にとって「見たことのない回転」**
4. 結果として **内積値は意味のない（学習されていない）値になり、attention 分布が崩壊**

**形式的可能性（数学）と実用的可能性（学習器の経験範囲）を区別する**のが本質的な答え。

---

## 解決法 — RoPE 拡張の 5 世代系譜

[[sources/yarn]] が整理した解決法の系譜（[[concepts/rotary-position-embeddings]] §「派生」でも詳述）：

```
PI (Chen 2023, Kaiokendev 2023)
    │ 位置インデックスを m → m/s で均等圧縮
    │ → 全次元を訓練分布内に押し込む
    │ ⚠️ 高周波情報を潰してしまう
    ▼
NTK-aware (bloc97 2023, Reddit)
    │ 基底変換 b → b' = b·s^(|D|/(|D|-2))
    │ → 高周波次元はほぼ保持、低周波次元を選択的にスケール
    │ ⚠️ 一部次元が「out-of-bound」に外挿される
    ▼
NTK-by-parts (bloc97 2023, GitHub) ← ターゲット補間の確立
    │ 波長 λ_i ごとに戦略を変える:
    │   λ < L (高周波): 補間しない（局所相対距離保持）
    │   λ ≥ L (低周波): 完全に補間（外挿せず）
    │   中間: ramp 関数 γ(r) で滑らかに遷移
    │
    ├─→ Dynamic NTK (emozilla 2023, Reddit)
    │       推論時に s = max(1, l'/L) を動的更新
    │       ファインチューン不要、Qwen 7B 採用
    │
    └─→ YaRN ([[sources/yarn]], 2023, ICLR 2024)
            = NTK-by-parts + attention 温度スケーリング
            √(1/t) = 0.1·ln(s) + 1
            Llama 2 を 4k → 128k に 400 ステップで拡張
```

### NTK-by-parts が本質的な洞察

ユーザの問題（低周波次元が訓練範囲外）を **波長ごとの戦略分岐** で解決：

- **波長 $\lambda < L$（高周波）**: 訓練中に十分回転を経験している → そのまま外挿可能
- **波長 $\lambda \geq L$（低周波）**: 訓練中に 1 回転すら見ていない → **線形補間で訓練分布内に押し込む**
- **中間**: ramp 関数 $\gamma(r)$ で滑らかに遷移、境界パラメータ $\alpha=1, \beta=32$（Llama）

これにより **「低周波次元が訓練分布外に出ない」** ことが保証され、高周波の局所情報も保持される。

---

## 実用的影響 — wiki の MLLM での実例

[[questions/large-scale-pretraining-series]] や [[questions/vit-dynamic-resolution-evolution]] とも関連する CV / MLLM 文脈：

### Qwen-VL 系列の長文脈外挿

| モデル | 訓練文脈 | 推論文脈 | 手法 | 結果 |
|---|---|---|---|---|
| [[entities/qwen2-vl|Qwen2-VL]] (2024) | 16K | **80K** | M-RoPE の位置 ID 値抑制 + YaRN 系 | 5× 外挿で安定 |
| [[entities/qwen2-5-vl|Qwen2.5-VL]] (2025) | 32K | (拡張) | MRoPE Aligned to Absolute Time | 動画時間グラウンディング SOTA |
| [[entities/qwen3-vl|Qwen3-VL]] (2025) | **256K** ネイティブ | **1M** YaRN 外挿 | **YaRN $s \approx 4$** | Needle-in-a-Haystack **1M で 99.5%** |

**Qwen3-VL の 1M YaRN 外挿は、本ページが解説したメカニズムを MLLM スケールで実用化した代表例**。256K → 1M の 4× 拡張で、本ページが議論した「低周波次元の分布外問題」を YaRN の波長別ターゲット補間で解決している。

### DINOv3 の axial RoPE + box jittering

[[entities/dinov3]] / [[sources/dinov3]] は CV で axial RoPE を採用するが、長距離外挿問題には **訓練時の対策（RoPE-box jittering、座標範囲を $[-s, s]$ にランダムスケール）** で対応：

- 訓練時に多様な「実効解像度」を見せることで、推論時の解像度変化に頑健化
- **YaRN の Dynamic Scaling と発想が近い**（訓練時 vs 推論時の違いはあるが、「事前に想定範囲を広げて学習させる」点で共通）

---

## まとめ — ユーザの問いへの直接的回答

### Q: 学習時の最大系列長が 1000 の場合、推論時に 1 番目と 2000 番目の相対関係はわからないのではないでしょうか？

**A: その通り、素の RoPE では事実上わからない**。

理由を 3 段階で整理：

1. **数式的には計算可能**: $\cos(1999\theta_i), \sin(1999\theta_i)$ は確定値として計算できる
2. **しかし学習器（$W_q, W_k$）は訓練分布外の回転角に対する適切な内積パターンを学習していない**: 特に **低周波次元（高い $i$、長い波長 $\lambda_i$）が訓練中に 1 回転すら経験していない**ため、その範囲外の回転角は完全に未学習
3. **結果として attention の内積値が壊れ、性能が崩壊**

これを解決するために：
- **PI**: 位置 1999 を $1999/2 = 999.5$ に圧縮（訓練分布内に押し込む）
- **NTK-aware**: 高周波は保持、低周波だけ圧縮
- **NTK-by-parts / YaRN**: 波長 $\lambda_i \geq L$ の次元のみ補間、$\lambda_i < L$ は外挿
- **Dynamic NTK / Dynamic-YaRN**: ファインチューン不要、推論時に動的にスケール

### 重要なメンタルモデル

- **RoPE の「任意位置で動く」性質**は **数式上の優位性**であって **実用的な外挿能力ではない**
- 学習可能 1D 位置埋め込みに対する **本質的優位性**は「補間不要・追加パラメータ不要・任意解像度に形式的に対応」
- **実用的外挿能力**は別途 YaRN 等の拡張で獲得する必要がある
- これが **wiki の [[concepts/rotary-position-embeddings]] が一見「あらゆる解像度で動く」と書きつつ「派生」セクションで YaRN を必要とする理由**

---

## 関連ページ

### 直接の前提

- [[concepts/rotary-position-embeddings]] — RoPE 概念ページ、5 世代の派生系譜
- [[sources/roformer]] — RoPE 原典（Su et al. 2021）、§4.5.5 で長文脈外挿の限界を著者自身が認める
- [[translations/roformer]] — 原典本文和訳

### 解決手法の論文

- [[sources/yarn]] — **本ページの分析の理論的基盤**。NTK-aware / NTK-by-parts / Dynamic NTK の系譜を学術的に統合した論文
- [[translations/yarn]] — YaRN 原典本文 + Appendix 和訳

### CV / MLLM での実用例

- [[entities/qwen3-vl]] / [[sources/qwen3-vl]] — **256K ネイティブ + 1M YaRN 外挿の代表**
- [[entities/qwen2-vl]] / [[sources/qwen2-vl]] — 16K → 80K 外挿、M-RoPE の位置 ID 抑制効果
- [[entities/qwen2-5-vl]] / [[sources/qwen2-5-vl]] — MRoPE Aligned to Absolute Time
- [[entities/dinov3]] / [[sources/dinov3]] — axial RoPE + RoPE-box jittering（CV での RoPE 外挿対策）

### 関連 question

- [[questions/vit-dynamic-resolution-evolution]] — ViT における解像度処理の進化、RoPE の役割と本問題の CV 視点での発展
- [[questions/large-scale-pretraining-series]] — 大規模事前学習 5 系列、RoPE 系の長文脈手法の位置付け
