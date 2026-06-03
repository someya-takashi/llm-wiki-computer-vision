---
type: source
source_path: raw/papers/YaRN_ Efficient Context Window Extension of Large Language Models.md
source_kind: paper
title: "YaRN: Efficient Context Window Extension of Large Language Models"
authors: [Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole]
year: 2023
venue: arXiv 2309.00071 / ICLR 2024
ingested: 2026-06-03
tags: [rope, context-extension, long-context, position-encoding, ntk-aware, attention]
translation: "[[translations/yarn]]"
---

# YaRN: Efficient Context Window Extension of Large Language Models（YaRN: 大規模言語モデルの効率的な文脈窓拡張）

> 原典: [[translations/yarn]] ・ `raw/papers/YaRN_ Efficient Context Window Extension of Large Language Models.md`
> 著者: Bowen Peng (Nous Research / Reddit `bloc97`), Jeffrey Quesnelle (Nous Research), Honglu Fan (EleutherAI / University of Geneva), Enrico Shippole
> 出典: arXiv:2309.00071（2023 年 8 月、ICLR 2024 採択）
> 公開コード・モデル: https://github.com/jquesnelle/yarn

## 一言まとめ

**[[sources/roformer|RoPE]] で事前学習された LLM の文脈窓を「事前学習データの 0.1% 未満」のファインチューニングで効率的に拡張する手法を提示し、Llama 2 を 4k → 128k に拡張して SOTA を達成した、RoPE 拡張系の決定的論文**。本論文は **Position Interpolation (PI) → NTK-aware → NTK-by-parts → Dynamic NTK → YaRN** の 5 世代に及ぶ技術系譜を整理し、最終的に「**NTK-by-parts 補間 + pre-softmax attention 温度スケーリング**」の組み合わせとしての YaRN を提案。**Qwen3-VL の 256K ネイティブ → 1M YaRN 外挿、Mistral の長文脈変種、多数のオープン MLLM の長文脈対応** の理論的基盤として、現代 LLM・MLLM の長文脈実用化を支える最重要原典の 1 つ。著者の **Bowen Peng は Reddit の `/u/bloc97`**（NTK-aware を最初に発見した本人）で、Nous Research / EleutherAI コミュニティから生まれた論文。

## 背景と問題意識

### なぜ RoPE 拡張が必要だったか

[[sources/roformer|RoPE]] 自体は「位置 $m$ に角度 $m\theta$ の回転を適用する」という連続値スキームのため、**形式的には任意位置に外挿可能** に見える。しかし実際には：

- LLaMA / Llama 2: 訓練文脈 **2k〜4k**
- 推論時に 8k、16k、64k を要求 → **訓練範囲を超えると性能が急激に劣化**
- RoFormer 論文（[[sources/roformer]] §4.5.5）自身が「長文脈外挿の挙動の理論的説明はない」と限界として認めている

これは **RoPE が訓練で見た位置の周波数空間に過適合**しているため。**未見の位置インデックスで内積に未知パターンが現れ、attention の挙動が崩壊**する。

### 5 世代の系譜（本論文の最重要貢献）

本論文は、コミュニティで分散していた長文脈手法の系譜を初めて学術的に整理：

```
PI (Chen 2023, Kaiokendev 2023)
    │ 位置インデックス全体を均等に圧縮 m → m/s
    ▼
NTK-aware (bloc97 2023, Reddit)
    │ 周波数を基底変換で再分配（高周波 ≒ 保持 / 低周波 ≒ スケール）
    ├─→ Dynamic NTK (emozilla 2023, Reddit)
    │       推論時に s を動的更新（ファインチューン不要）
    │
    └─→ NTK-by-parts (bloc97 2023, GitHub)
            波長 λ_d を考慮、高周波は補間せず・低周波は完全補間
            │
            │ + pre-softmax attention 温度スケーリング
            ▼
        YaRN (本論文)
```

### Reddit / GitHub から ICLR 2024 へ

本論文の特徴は **NTK-aware / Dynamic NTK / NTK-by-parts** が元々 **Reddit 投稿と GitHub プルリクエスト**で公開された手法だった点。著者 Bowen Peng は `bloc97` という Reddit ハンドルで NTK-aware と NTK-by-parts を最初に発見・公開。本論文はこれら **コミュニティ駆動の発見を学術的に整理・統合し、YaRN として完成**させた。

## 提案手法 / 主張：YaRN の核心

### Position Interpolation (PI) の出発点

Chen et al. (2023) と Kaiokendev (2023) が並行発見：

$$f^{\prime}_{\mathbf{W}}(\mathbf{x}_{m}, m, \theta_{d}) = f_{\mathbf{W}}\left(\mathbf{x}_{m}, \dfrac{mL}{L^{\prime}}, \theta_{d}\right)$$

つまり「**位置インデックスを $L/L^{\prime}$ で圧縮**して訓練範囲内に押し込む」。**スケール係数 $s=L^{\prime}/L$** を定義（後の全手法で使われる重要表記）。

**長所**: 単純、ファインチューンと組み合わせて効く
**欠点**: 全次元を均等に圧縮するため **高周波情報が失われる**（細かい相対距離を区別する能力が低下）

### NTK-aware: 周波数の選択的スケーリング

Neural Tangent Kernel 理論から：「**ネットワークは低次元入力では高周波情報を学習しにくい**」。RoPE は 1D 位置を多次元複素ベクトルに展開する Fourier Feature の特別ケースなので、**RoPE を均等に引き伸ばすと高周波が潰れる**。

解決策：**基底 $b$ を変換**して周波数分布を再配置：

$$b^{\prime} = b\cdot s^{\frac{|D|}{|D|-2}}, \quad h(\theta_{d}) = {b^{\prime}}^{-2d/|D|}$$

これで **高周波次元 ≒ 元のスケール / 低周波次元 ≒ PI 並みのスケール** で、補間圧力を分散。

**Code Llama** が「NTK-aware」を **$b$ を 1M に手動設定**して採用（[^31]、本論文発表直前）。

**欠点**: 一部次元が「out-of-bound」に外挿される（理論的補間ではなく）→ ファインチューンすると PI に劣る。

### NTK-by-parts: 波長を意識したターゲット補間

「全次元を均等に扱う」を捨て、**各次元の波長 $\lambda_d = 2\pi b^{2d/|D|}$** ごとに戦略を変える：

- **波長 $\lambda < L$**（高周波）: 補間しない（局所相対距離を保持）
- **波長 $\lambda > L$**（低周波）: 完全に補間（外挿せず）
- 中間: ramp 関数 $\gamma(r)$ で滑らかに遷移

ratio $r(d) = L/\lambda_d$ を導入し、パラメータ $\alpha, \beta$ で境界を制御（Llama は $\alpha=1, \beta=32$）：

$$h(\theta_d) = (1-\gamma(r(d)))\frac{\theta_d}{s} + \gamma(r(d))\theta_d$$

これが「**ターゲット（targeted）補間**」の典型で、**「ブラインド（blind）補間」（PI、NTK-aware）と対立**。

### Dynamic NTK: 推論時の動的スケーリング

ファインチューンなしで使う場合の重要な発見：**スケール $s$ を固定するのではなく、各 forward-pass で動的更新**：

$$s = \max(1, l^{\prime}/L)$$

ここで $l^{\prime}$ は現在の系列長。これで：
- $l^{\prime} \leq L$ では $s=1$（元の RoPE と同じ）
- $l^{\prime} > L$ では $s$ が漸進的に増加（段階的劣化）

**Qwen-7B** が初期に「Dynamic NTK」を採用（本論文 [^2]）。

**重要な注意**: kv キャッシングと組み合わせる場合、**RoPE 適用前に kv をキャッシュ**しなければならない（$s$ 変更で全トークンの RoPE 埋め込みが変わる）。

### YaRN: NTK-by-parts + Attention 温度スケーリング

本論文の中心的貢献は **「pre-softmax attention に温度 $t$ を導入」** という追加の革新：

$$\text{softmax}\left(\dfrac{\mathbf{q}_m^T \mathbf{k}_n}{t\sqrt{|D|}}\right)$$

実装上の美点は、**RoPE 埋め込み自体に $\sqrt{1/t}$ を掛けるだけで等価**（query/key 両方が同じ係数でスケールされるため、内積は $1/t$ 倍される）。**コード変更不要・推論コスト ゼロ**。

LLaMA / Llama 2 の推奨値：

$$\sqrt{\frac{1}{t}} = 0.1\ln(s) + 1$$

この式は LLaMA 7b/13b/33b/65b で経験的にフィッティングしたもの。**Llama 2 にも一般化**したことから「**$t$ の普遍性**」が示唆される。

YaRN = **NTK-by-parts 補間 + attention 温度スケーリング** の組み合わせ。

## 実験結果と知見

### Llama 2 7B/13B を 128k に拡張

| データセット | 文脈サイズ | 訓練 | 結果 |
|---|---|---|---|
| Proof-pile | 4k → 8k | 400M トークン | PI (1B): 8k で perplexity 3.34 → YaRN: 3.35（同等、データ 1/2.5） |
| Proof-pile | 4k → 64k ($s=16$) | 400 ステップ | 64k で perplexity **2.42**（Code Llama 2.55、Together PI は爆発） |
| Proof-pile | 4k → 128k ($s=32$) | +200 ステップ | **128k で perplexity 2.37**（Code Llama 2.71 を凌駕） |

特筆すべきは **$s=32$ モデルが訓練時 64k のみで 128k に外挿できた**こと。これは「train short, test long」を実証。

### Passkey 検索（B.2）

- **YaRN 7B/13B $s=32$ の 128k モデル**: 全文脈窓で **>99% 精度**
- Code Llama 7B も善戦（112k で 94.3%）だが、YaRN $s=32$ 7B は 128k で 99.4%

### 標準 LLM ベンチマークでの劣化（Table 3）

ARC-c / HellaSwag / MMLU / TruthfulQA で評価：

- **YaRN ($s=16, s=32$) は Llama 2 ベースラインから最小限の劣化**
- Code Llama (NTK-aware) は **ARC-c 53.1→39.9、MMLU 43.8→31.1 と大幅劣化** ← 「out-of-bound 外挿」の代償
- $s=16 \to s=32$ の追加拡張は **平均 0.49% しか劣化しない**

### 計算効率

- **訓練ステップ**: PI/NTK 1000+ ステップ vs **YaRN 400 ステップ**（2.5× 削減）
- **訓練データ**: Code Llama 4B トークン vs **YaRN 400M**（10× 削減）
- 元の事前学習データの **約 0.1%** で済む

### Mistral 7B v0.1 への拡張（B.4）

- $s=8$ で 1000 ステップ → 64k モデル
- $s=16$ で追加 500 ステップ → 128k モデル
- スライディング・ウィンドウ attention を無効化（YaRN の長距離注意と競合するため）
- **MistralLite (NTK-aware, $\theta=1$M) を凌駕**

### Dynamic-YaRN（B.3）: ファインチューンなしでも有効

- 元の Llama 2（ファインチューンなし、文脈 4k）に Dynamic Scaling を適用
- **Dynamic-YaRN > Dynamic-PI**（perplexity 比較）
- 文脈窓爆発を防ぎ「優雅な劣化」を実現

## 限界・批判的視点

### 著者が触れていない / 暗黙の前提

1. **RoPE 専用**: 本論文の手法は RoPE で訓練されたモデルにのみ適用可能。学習可能位置埋め込み（オリジナル ViT 等）、ALiBi、T5 Relative Bias には適用不可。**Vision モデルでの応用は DINOv3（axial RoPE）/ Qwen-VL 系（M-RoPE）など RoPE 採用モデルに限定**
2. **「Effective context size」と perplexity の乖離**（B.2 で著者自身が認める）: Code Llama 13B は 100k で perplexity が増加するが、passkey 検索は 128k でも 99.4%。perplexity が長文脈評価の万能指標でないことを示唆
3. **温度スケーリングの理論的説明が弱い**: Eq. 22 の $0.1\ln(s)+1$ は経験的フィッティングで、**「なぜ機能するか」の数学的説明は与えられていない**。Llama 2 への一般化は「普遍性かもしれない」という仮説に留まる

### 本 wiki の視点からの追加批判

4. **コンセプチュアルな革新の希薄さ**: 4 つの先行手法（PI, NTK-aware, Dynamic NTK, NTK-by-parts）の整理 + 新規 1 要素（温度スケーリング）の組み合わせ。**YaRN そのものは増分的改良**で、本論文の真の貢献は「整理と統合」
5. **コミュニティ駆動の発見の学術化**: NTK-aware（bloc97 Reddit 投稿）、Dynamic NTK（emozilla Reddit 投稿）、NTK-by-parts（bloc97 GitHub PR）は**いずれもピアレビューを経ない**コミュニティ発見だった。本論文はこれらを **「事後的に学術形式に整える」役割**。これは現代 ML 研究の生態系を象徴する事例
6. **Vision モデルへの直接適用は限定的**: 本論文の評価は全て言語モデル（Llama 2、Mistral）。**Vision Transformer での YaRN 直接適用の論文はまだ稀**。Vision では DINOv3 の axial RoPE のように **RoPE 自体の 2D 化** が先行
7. **MLLM での非自明な使用**: Qwen3-VL の 1M YaRN 外挿は **本論文の YaRN $s=32$ 設定の延長線上**だが、Qwen-VL 独自の M-RoPE / Interleaved MRoPE との組み合わせ方は本論文では未論証。**MLLM 文脈での YaRN ベストプラクティスは依然エンジニアリング知識**
8. **OpenAI / Anthropic 系商用モデルとの関係**: GPT-4 や Claude の長文脈（128k〜1M）の実装は非公開。YaRN が業界標準かは不明（ALiBi 系、SSM 系、その他の手法を使う可能性）
9. **「効果的文脈長」の評価の貧しさ**: 本論文の評価は perplexity と passkey 検索のみ。**Needle-in-a-Haystack（複数の針）、RULER、LongBench 等の現代的長文脈ベンチマーク**は本論文後に登場。Qwen3-VL の 1M Needle 99.5% などの評価は YaRN の正当性を後追いで実証

## 用語と略称

- **YaRN**: **Y**et **a**nother **R**oPE extensio**N** method（本論文の中心手法）
- **PI**: **P**osition **I**nterpolation（位置補間、Chen 2023）
- **NTK**: **N**eural **T**angent **K**ernel（深層 NN の関数空間挙動を解析する理論枠組み）
- **NTK-aware interpolation**: 周波数を基底変換で再分配する補間（bloc97 2023、Reddit）
- **NTK-by-parts interpolation**: 波長ごとに補間戦略を変える（bloc97 2023、GitHub）
- **Dynamic NTK / Dynamic Scaling**: 推論時に $s$ を動的更新（emozilla 2023、Reddit）
- **Dynamic-YaRN**: YaRN を Dynamic Scaling と組み合わせた推論時手法
- **scale factor $s$**: $L^{\prime}/L$、拡張比率（PI 以降の全手法で共通）
- **wavelength $\lambda_d = 2\pi b^{2d/|D|}$**: 次元 $d$ で完全な回転を行うのに必要なトークン長
- **blind interpolation**: 波長を気にしない補間（PI、NTK-aware）
- **targeted interpolation**: 波長を考慮する補間（NTK-by-parts、YaRN）
- **ramp function $\gamma(r)$**: NTK-by-parts の境界遷移関数
- **$\alpha, \beta$**: NTK-by-parts の境界パラメータ（Llama では 1, 32）
- **pre-softmax temperature $t$**: YaRN が attention 重みに導入する温度
- **PG19**: 訓練に使った長文書データセット（Project Gutenberg books）
- **Proof-pile**: 評価用の数学証明データセット
- **GovReport**: 政府報告書の長文書データセット
- **Passkey Retrieval**: 大量の無意味テキスト中の 5 桁数字を検索するタスク（Mohtashami & Jaggi 2023）
- **Flash Attention 2**: YaRN と直接互換な高速 attention 実装（Dao 2023）
- **bloc97**: Reddit ハンドル、NTK-aware / NTK-by-parts の最初の発見者、本論文の Bowen Peng
- **emozilla**: Reddit ハンドル、Dynamic NTK の最初の発見者、本論文の Jeffrey Quesnelle

## CV / MLLM における意義（wiki 既存ページとの接続）

YaRN は NLP 論文だが、本 wiki の **MLLM（マルチモーダル LLM）の長文脈対応** の理論的基盤として極めて重要。

### YaRN を活用する MLLM

- **[[entities/qwen3-vl|Qwen3-VL]]**（Alibaba, 2025、[[sources/qwen3-vl]]）: **256K ネイティブ → 1M YaRN 外挿**。**Needle-in-a-Haystack で 256K 100% / 1M 99.5%** という驚異的な長文脈性能。YaRN $s=32$ の延長線上で、Llama 2 で 128k を達成した技法を MLLM スケールに適用
- **[[entities/qwen2-5-vl|Qwen2.5-VL]]**（Alibaba, 2025）: 32K 文脈で YaRN 系の拡張を活用
- **[[entities/qwen2-vl|Qwen2-VL]]**（Alibaba, 2024）: 学習 16K → 推論 80K の外挿は M-RoPE の位置 ID 抑制効果 + YaRN 系技法の組み合わせ

### RoPE 系の長文脈拡張の最重要原典

[[concepts/rotary-position-embeddings|RoPE]] の概念ページの「派生」セクションで言及される **NTK-aware RoPE / YaRN / LongRoPE** の中心が本論文。LongRoPE（2024）も YaRN の発展形として理解できる。

### Vision での適用は限定的

本論文の対象は LLM だが、**Vision での RoPE 採用は DINOv3（axial RoPE）/ SAM 2（memory 2D-RoPE）/ Qwen-VL 系（M-RoPE）** などに広がる。これらモデルで「長系列外挿」が必要な場合（高解像度推論、動画長さ拡張）に YaRN の発想を借りる動きがある：

- **DINOv3 の RoPE-box jittering**（[[sources/dinov3]]）: 訓練時に座標範囲をランダムスケール。**YaRN の Dynamic Scaling と発想が近い**
- **Qwen2-VL の長文脈外挿**: 学習 16K → 推論 80K は YaRN 風技法に依拠

## 関連ページ

### 直接の派生・前提

- [[concepts/rotary-position-embeddings]] — RoPE 概念ページ。YaRN は「派生」セクションの中心
- [[sources/roformer]] — **RoPE 原典**。YaRN が拡張する対象
- [[translations/yarn]] — 原典の本文 + Appendix 和訳

### YaRN を活用する MLLM

- [[entities/qwen3-vl]] / [[sources/qwen3-vl]] — **256K ネイティブ + 1M YaRN 外挿**、YaRN の最大規模応用例
- [[entities/qwen2-5-vl]] / [[sources/qwen2-5-vl]] — 32K で YaRN 系拡張
- [[entities/qwen2-vl]] / [[sources/qwen2-vl]] — 16K→80K 外挿、M-RoPE + YaRN 系
- [[entities/internvl-3]] / [[sources/internvl-3]] — V2PE（変動視覚位置エンコーディング）、概念的に NTK-by-parts に近い

### RoPE 系の他のモデル

- [[entities/dinov3]] / [[sources/dinov3]] — axial RoPE、Vision での RoPE 採用
- [[entities/sam-2]] / [[sources/sam-2]] — memory cross-attention で 2D-RoPE

### 長文脈の議論

- [[questions/vit-dynamic-resolution-evolution]] — ViT の解像度（=系列長）外挿の進化。Vision における YaRN の役割を整理
- [[questions/large-scale-pretraining-series]] — 大規模事前学習 5 系列の分類、RoPE と YaRN の位置付け

### 上位概念

- [[concepts/vision-transformer]] — ViT、長系列処理の対象アーキテクチャ
- [[concepts/foundation-model]] — 基盤モデル全体
