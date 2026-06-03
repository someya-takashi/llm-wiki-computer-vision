---
type: source
source_path: raw/papers/RoFormer_ Enhanced Transformer with Rotary Position Embedding.md
source_kind: paper
title: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
authors: [Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu]
year: 2021
venue: arXiv 2104.09864 / Neurocomputing 2024
ingested: 2026-06-03
tags: [position-encoding, rope, transformer, nlp, attention]
translation: "[[translations/roformer]]"
---

# RoFormer: Enhanced Transformer with Rotary Position Embedding（RoFormer: 回転位置埋め込みを備えた強化版 Transformer）

> 原典: [[translations/roformer]] ・ `raw/papers/RoFormer_ Enhanced Transformer with Rotary Position Embedding.md`
> 著者: Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu（Zhuiyi Technology Co., Ltd., 深圳）
> 出典: arXiv:2104.09864（2021 年 4 月初版、Neurocomputing 誌 2024 掲載）
> 公開コード: https://github.com/ZhuiyiTechnology/roformer
> Huggingface 統合: https://huggingface.co/docs/transformers/model_doc/roformer

## 一言まとめ

**Transformer の self-attention に「位置情報を回転行列の乗算として注入する」新しい位置エンコーディング手法 RoPE（Rotary Position Embedding, 回転位置埋め込み）を提案した、現代 LLM・MLLM の事実上の標準となった重要論文**。位置 $m$ にあるトークンの query/key ベクトルを角度 $m\theta$ だけ回転させると、内積 ${\boldsymbol q}_m^{\intercal}{\boldsymbol k}_n$ には相対位置 $m-n$ のみが自然に現れる。**加算ではなく乗算**である点が革新。発表時は中国語法律文書のような長文タスク向けの改善として出されたが、後に **LLaMA / GPT-NeoX / PaLM / Mistral / Qwen / DeepSeek など主要 LLM のすべてが採用**、さらに **DINOv3 / Qwen2-VL / SAM 2 などの Vision/Multimodal モデルでも標準採用**され、Computer Vision における**任意解像度対応**の決定的技法となった。

## 背景と問題意識

### Transformer は位置非依存（position-agnostic）

オリジナル Transformer（Vaswani et al., 2017）の self-attention は順序を全く考慮しない。トークン集合に対する「集合変換」であり、入力順序を入れ替えても出力が同じになる。このため位置情報を別途注入する機構が必要。

> **補足: なぜ位置情報が必要か** — 自然言語では「犬が人を噛む」と「人が犬を噛む」は明確に意味が違うが、トークン集合としては同じ。位置情報を Transformer に与えないと両者を区別できない。画像でも「左上のパッチ」と「右下のパッチ」を区別するため、ViT で同様の問題が生じる。

### 既存の位置エンコーディングの 4 系統と限界

論文の §2 は既存手法を体系的に整理する：

1. **絶対位置・正弦波（Vaswani et al., 2017）**: 位置を `sin(k/10000^(2t/d))` で生成、入力埋め込みに **加算**。学習パラメータ不要だが、attention で相対位置を直接活用できない。
2. **絶対位置・学習可能（BERT 系）**: 各位置に学習可能ベクトル。シンプルだが**系列長が学習時に固定される**問題。
3. **相対位置・バイアス型（Shaw 2018, T5, Swin）**: attention スコアに相対距離依存のバイアス $b_{i,j}$ を加算。相対距離を直接モデル化できるが、実装が複雑で attention 計算コストが上がる。
4. **相対位置・分解型（Transformer-XL, DeBERTa）**: ${\boldsymbol q}_m^{\intercal}{\boldsymbol k}_n$ を 4 項に分解し、絶対位置 ${\boldsymbol p}_m, {\boldsymbol p}_n$ を相対版 $\tilde{\boldsymbol p}_{m-n}$ で置換。

**著者らが指摘した共通の限界**: これらすべては「位置情報を文脈表現に加算する」という発想に基づく。**この加算的な性質ゆえに、線形 self-attention（Performer 等）と両立しない**。線形 attention は query/key の非線形変換 $\phi, \varphi$ を適用して内積を分離するが、位置エンコーディングを加算するとこの分離が崩れる。

> **補足: 線形 attention とは** — 通常の self-attention は $\mathbb{O}(N^2)$ の計算量がボトルネックだが、$\operatorname{sim}({\boldsymbol q}, {\boldsymbol k}) = \phi({\boldsymbol q})^{\intercal}\varphi({\boldsymbol k})$ のように分離可能な形にすると、行列乗算の結合性で $\mathbb{O}(N)$ に削減できる（Performer, Linear Transformer）。RoPE が「ノルム不変な乗算的回転」である点はこの線形 attention との両立に決定的。

## 提案手法 / 主張：RoPE の核心

### 出発点：「内積が相対位置のみの関数」という制約

論文の §3.1 は明確な制約から出発する：query $f_q({\boldsymbol x}_m, m)$ と key $f_k({\boldsymbol x}_n, n)$ の内積が、相対位置 $m-n$ のみに依存する関数 $g$ で書けることを要求する：

$$\langle f_q({\boldsymbol x}_m, m), f_k({\boldsymbol x}_n, n)\rangle = g({\boldsymbol x}_m, {\boldsymbol x}_n, m-n)$$

これは「あらゆる絶対位置 $m, n$ について、attention は $m-n$ にしか依存してほしくない」という素朴な要請。この制約から $f_q, f_k$ の形を**逆算**するのが論文の核心アプローチ。

### 2D での解：複素数 $e^{im\theta}$ の乗算

$d=2$ の単純な場合、2D ベクトルを複素数として扱うと（${\boldsymbol x} = x_1 + i x_2$）、解は驚くほど美しい：

$$f_q({\boldsymbol x}_m, m) = ({\boldsymbol W}_q {\boldsymbol x}_m) \cdot e^{im\theta}, \quad f_k({\boldsymbol x}_n, n) = ({\boldsymbol W}_k {\boldsymbol x}_n) \cdot e^{in\theta}$$

つまり「位置 $m$ のベクトルに $e^{im\theta}$ を掛ける」だけ。内積を取ると：

$$\operatorname{Re}[({\boldsymbol W}_q {\boldsymbol x}_m) ({\boldsymbol W}_k {\boldsymbol x}_n)^{*} e^{i(m-n)\theta}]$$

となり、**相対位置 $m-n$ が自然に現れる**。複素数の乗算は実行列では 2D 回転に対応するため、これは「query/key を位置に応じた角度で回転させる」操作と等価。

### 一般次元への拡張：ブロック対角の回転行列

d 次元（偶数）への一般化は単純で、$d/2$ 個の 2D 部分空間に分け、各部分空間で異なる角度 $\theta_i = 10000^{-2(i-1)/d}$（正弦波エンコーディングと同じ系列）で回転：

$$f_{\{q,k\}}({\boldsymbol x}_m, m) = {\boldsymbol R}^d_{\Theta, m} {\boldsymbol W}_{\{q,k\}} {\boldsymbol x}_m$$

ここで ${\boldsymbol R}^d_{\Theta, m}$ はブロック対角の回転行列で、$i$ 番目の 2×2 ブロックが角度 $m\theta_i$ の回転。**この行列は直交行列で**、ベクトルのノルムを保つ。

### 効率的実装：スパース性を活用

論文 §3.4.2 が示すように、回転行列は非常にスパースなので、**実装では明示的な行列乗算を行わず、要素ごとの掛け算とインデックス入れ替えで済む**：

```
RoPE(x)[2i]   = x[2i]   * cos(m·θ_i) - x[2i+1] * sin(m·θ_i)
RoPE(x)[2i+1] = x[2i+1] * cos(m·θ_i) + x[2i]   * cos(m·θ_i)
```

これは現在の RoPE 実装すべての標準形。

### 3 つの核心的性質

1. **相対位置の自然な出現**: 内積が ${\boldsymbol R}^d_{\Theta, n-m}$ という相対位置に依存する 1 つの行列だけで書ける（${\boldsymbol R}^d_{\Theta, n-m} = ({\boldsymbol R}^d_{\Theta, m})^{\intercal}{\boldsymbol R}^d_{\Theta, n}$ から直接導出）
2. **長期減衰（Long-term decay）**: $\theta_i = 10000^{-2i/d}$ という選択により、相対距離が大きくなるほど内積の上界が減衰する（§3.4.3、図 2）。**「遠いトークンは関係が薄い」という直観と一致**
3. **線形 attention との両立**: RoPE は乗算でノルム保存なので、線形 attention の非負関数 $\phi, \varphi$ の**出力に回転行列を掛ける**だけで組み合わせられる（§3.3）

## 実験結果と知見

### 4 つの NLP タスクでベースライン超え

| 実験 | データ | 比較 | 結果 |
|---|---|---|---|
| **§4.1 機械翻訳** | WMT 2014 英独 | Transformer-base | BLEU 27.3 → **27.5** |
| **§4.2 BERT 事前学習** | BookCorpus + Wiki | BERT-base | **MLM 損失が一貫して低い、収束が速い**（図 3 左） |
| **§4.3 GLUE ファインチューン** | 6 タスク | BERT | **3/6 タスクで上回る**（MRPC, STS-B, QQP）。SST-2/QNLI/MNLI では BERT が勝つ — 全勝ではない |
| **§4.4 Performer 線形 attention** | Enwik8 char-LM | Performer | RoPE 付与で収束加速 + 低損失（図 3 右） |
| **§4.5 中国語長文（CAIL2019-SCM）** | 法律文書 | BERT, WoBERT | 512 で同等、**1024 で WoBERT を +1.5% で上回る**。系列長拡張に強い |

### 重要な知見

- **GLUE で全勝でなかった**点は誠実な報告。RoPE は long-text タスクで特に効くが、短文で常に勝つわけではない（後の世代で他の改良と組み合わせて初めて全面的優位に）
- **長期減衰の理論的保証**は重要な貢献。学習可能な相対位置バイアスと違い、$\theta_i = 10000^{-2i/d}$ という設定だけで「遠いトークンは弱い影響」が数学的に証明される
- **線形 attention との両立は破壊的**: Transformer の二次計算量問題を解決する Performer/Linear Transformer に位置情報を入れる標準的方法を提供した

## 限界・批判的視点（著者自身が認めるもの）

§4.5.5 で著者が自ら認める限界：

1. **理論と実験のギャップ**: 数学的に綺麗な定式化と長期減衰の証明はあるが、**なぜ既存手法より速く収束するか、なぜ長文で特に強いかの説明は欠ける**。長期減衰は既存の正弦波・相対位置エンコーディングも持つ性質なので、それだけで RoPE の優位を説明できない
2. **事前学習コスト**: Transformer 基盤の上に構築されるため、検証に大規模ハードウェアが必要
3. **絶対位置の明示性**: 「文書の冒頭か末尾か」を強く区別したい用途では、純粋な相対位置エンコーディングは弱い面がある（後の Vision での研究で再認識）

### この wiki の視点からの追加批判

4. **GLUE では BERT に負けるタスクもある**（SST-2 で -2.8、MNLI で -4.4）。位置エンコーディングだけ替えれば常に勝てるわけではない
5. **発表時は影響が見えにくかった**: 2021 年 4 月の arXiv 初版時点ではあまり注目されず、Su が中国の Zhuiyi Technology という比較的小さな企業の研究者であったこと、ブログ記事 "Transformer 升级之路" で先に提案されていたことなど、論文の経路が NLP 主流の発表チャネルと違ったため、英語圏への浸透が遅れた
6. **長期外挿（context length extrapolation）の限界**: 訓練時の系列長を大きく超えると性能が劣化する問題は本論文ではほとんど触れられず、後に **NTK-aware RoPE / YaRN / LongRoPE** などの拡張手法で解決された

## 用語と略称

- **RoPE**: **Ro**tary **P**osition **E**mbedding（回転位置埋め込み）— 本論文の中心概念
- **RoFormer**: **Ro**tary **Former** — RoPE を組み込んだ Transformer の名称
- **PLM**: Pre-trained Language Model（事前学習済み言語モデル）
- **MLM**: Masked Language Modeling（マスク言語モデリング、BERT の事前学習目的）
- **LM**: Language Modeling（自己回帰的言語モデリング、GPT 系）
- **BPE**: Byte Pair Encoding（バイトペア符号化、サブワード・トークン化）
- **GLUE**: General Language Understanding Evaluation（NLP 標準ベンチマーク集）
- **MRPC / SST-2 / QNLI / STS-B / QQP / MNLI**: GLUE 内の個別タスク
- **CAIL2019-SCM**: Chinese AI and Law 2019 Similar Case Matching（中国法律文書類似マッチング・データセット）
- **WoBERT**: Word-based BERT for Chinese（中国語向けの単語ベース BERT、著者 Su の別作品）
- **NEZHA**: 華為製の中国語事前学習済み Transformer
- **NTK-aware RoPE / YaRN / LongRoPE**: 本論文後に提案された RoPE の長文外挿改良（本論文には未記載、wiki 別ページ [[concepts/rotary-position-embeddings]] 参照）
- **Performer**: 線形 attention を実装した Transformer 変種（§4.4 で RoPE と組み合わされる）
- **正弦波エンコーディング（sinusoidal encoding）**: Vaswani 2017 のオリジナル位置エンコーディング、$\sin(k/10000^{2t/d})$
- **相対位置エンコーディング**: query/key の絶対位置でなく相対距離 $m-n$ で attention に影響を与える方式
- **長期減衰（long-term decay）**: 相対距離が大きくなると attention 寄与が単調に減衰する性質
- **直交行列（orthogonal matrix）**: 転置が逆行列に等しい行列、ベクトルのノルムを保つ

## CV における意義（wiki 既存ページとの接続）

RoFormer 自身は NLP 論文だが、提案された RoPE は CV における**任意解像度対応**の決定的技法となった。本 wiki にすでに登録済みのモデル・概念での具体例：

### 採用された Vision モデル

- **[[entities/dinov3|DINOv3]]**（Meta, 2025、[[sources/dinov3]]）: **axial RoPE**（特徴量の半分を X 軸の回転、もう半分を Y 軸の回転に割り当て）+ **RoPE-box jittering**（正規化座標を $[-s, s]$ にランダム・スケール、$s \in [0.5, 2]$）。**256² で訓練したモデルが 4096² でも安定動作**
- **[[entities/sam-2|SAM 2]]**（Meta, 2024、[[sources/sam-2]]）: **memory cross-attention でのみ 2D-RoPE を採用**、画像エンコーダ（Hiera）側は絶対位置埋め込み。「memory への spatial 一貫性が必要だが画像エンコーダはシンプルに保ちたい」という設計判断
- **[[entities/qwen2-vl|Qwen2-VL]]**（Alibaba, 2024、[[sources/qwen2-vl]]）: **M-RoPE**（Multimodal RoPE）として、回転位置埋め込みを **temporal / height / width の 3 成分に分解**。テキストは 1D-RoPE 等価、画像は temporal 固定、動画は temporal 増分。**学習 16K → 推論 80K トークンまで頑健に外挿**
- **[[entities/qwen2-5-vl|Qwen2.5-VL]]**（Alibaba, 2025、[[sources/qwen2-5-vl]]）: **MRoPE Aligned to Absolute Time** — Qwen2-VL の M-RoPE で temporal ID をフレーム番号でなく **絶対秒数**に揃え、FPS 非依存な時間整合を学習。Charades-STA mIoU 50.9 SOTA
- **[[entities/qwen3-vl|Qwen3-VL]]**（Alibaba, 2025、[[sources/qwen3-vl]]）: **Interleaved MRoPE** — t/h/w 成分を埋め込み次元にわたって**均一に交互配置**することで、Qwen2-VL/2.5-VL の塊状分割による周波数スペクトル不均衡を解消。**256K ネイティブ / 1M YaRN 外挿**

### CV における RoPE の決定的優位（[[concepts/rotary-position-embeddings]] より）

1. **任意解像度対応**: 学習可能絶対位置埋め込みは「位置 1〜196 用のベクトル」のみ訓練したきり拡張不可。RoPE は位置 $m$ を連続値として角度 $m\theta$ を計算するだけなので、**未見の解像度でも動作**
2. **パラメータ追加ゼロ**: モデルサイズに影響しない
3. **モダリティ非依存**: NLP/CV/音声で同じ実装が使える
4. **動画・3D・マルチモーダルへの自然な拡張**: 2D-RoPE（axial）/ 3D-RoPE / M-RoPE（temporal+spatial 分解）など、軸の追加で次元拡張が容易

## 関連ページ

- [[concepts/rotary-position-embeddings]] — RoPE の概念解説。本論文がその「原典」。NLP 採用モデル（LLaMA, Mistral, PaLM, Qwen, DeepSeek 等）と CV 採用モデル（DINOv3, SAM 2, Qwen-VL 系）を網羅
- [[concepts/vision-transformer]] — RoPE が改善する対象アーキテクチャ
- [[sources/dinov3]] — Vision で axial RoPE を本格採用、box jittering で解像度頑健性を強化
- [[sources/sam-2]] — memory attention で 2D-RoPE を採用、画像エンコーダ側は絶対位置埋め込みのみという設計選択
- [[sources/qwen2-vl]] — M-RoPE（temporal/height/width 3 成分）を導入し ViT 側も 2D-RoPE で任意解像度対応にした論文
- [[sources/qwen2-5-vl]] — M-RoPE を絶対時間に整合させた論文
- [[sources/qwen3-vl]] — Interleaved MRoPE + テキスト・ベース時間整合を導入した最新版
- [[entities/dinov3]] / [[entities/sam-2]] / [[entities/qwen2-vl]] / [[entities/qwen2-5-vl]] / [[entities/qwen3-vl]] — RoPE 採用モデル

### 本論文がスコープ外とした関連話題（wiki で個別管理）

- **NTK-aware RoPE / YaRN / LongRoPE**: 訓練時系列長を超える外挿の改良。本論文は触れない
- **画像 SSL 系の純粋 RoPE 評価**: 本論文は NLP のみ。Vision での包括的検証は DINOv3 や RoPE-ViT で行われる
- **画像生成系での RoPE**: Stable Diffusion 系（[[entities/sdxl]]）は学習可能埋め込みが標準、RoPE 採用は SD3 や FLUX.1 などの DiT 系で本格化（本 wiki では未 ingest 領域）
