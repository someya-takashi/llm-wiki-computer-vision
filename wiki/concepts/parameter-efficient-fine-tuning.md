---
type: concept
aliases: [PEFT, PETL, Parameter-Efficient Fine-Tuning, Parameter-Efficient Transfer Learning, パラメータ効率的ファインチューニング]
tags: [fine-tuning, transfer-learning, foundation-model, lora, adaptformer, bitfit, vpt, adapter, low-rank]
related: [[concepts/foundation-model]], [[concepts/vision-transformer]], [[concepts/semi-supervised-learning]], [[entities/clip]], [[entities/dinov2]]
sources: [[sources/revisiting-ssl-foundation-models]]
updated: 2026-05-28
---

# Parameter-Efficient Fine-Tuning（PEFT, パラメータ効率的ファインチューニング）

## 一言で

**事前学習済みの基盤モデルを下流タスクに適応させる際、モデル全体ではなく一部のパラメータだけを更新する手法群**。Full fine-tuning（全パラメータ更新）と Linear probing（最終層のみ）の中間に位置し、両者の弱点を回避する：

- Full fine-tuning は計算コストが高く、事前学習で得た汎化能力を破壊しがち
- Linear probing は表現力に限界があり、難タスクで性能が伸びない
- **PEFT は「最小の更新で最大の適応」を狙う**

NLP 領域では LoRA（Hu et al., 2021）が大規模言語モデル時代の標準となり、CV 領域でも 2022 年以降 LoRA、AdaptFormer、Visual Prompt Tuning（VPT）等が広く使われている。

## なぜ PEFT が必要か

### 1. 計算コスト・メモリ効率

[[entities/dinov2]] ViT-L（300M+ パラメータ）、CLIP ViT-L、SAM ViT-H 等の VFM を full fine-tune するには数十 GB の GPU メモリと長時間が必要。PEFT なら数百万パラメータの更新で済むため、消費者向け GPU でも fine-tune 可能。

### 2. 破滅的忘却（catastrophic forgetting）の回避

Full fine-tuning は事前学習で得た汎化能力を意図せず破壊することがある。特に下流タスクのデータが少ない場合に深刻。PEFT は元の重みを凍結したまま薄い「調整層」だけを変えるので、事前学習の知識を保持する。

### 3. ストレージ効率

タスクごとに full fine-tune した重みを保存すると、モデルサイズ × タスク数の容量が必要。PEFT なら「凍結ベース重み + 小さな PEFT モジュール」だけ保存すれば良く、数百タスクを 1 ベース重み + 小さな差分の集合として管理できる。

### 4. SSL/SeSL との相性

[[sources/revisiting-ssl-foundation-models]] は「**ラベルあきが少ない SeSL シナリオでは、Full fine-tuning より PEFT の方が顕著に優れる**」ことを実証した。これは PEFT が「少データでの過学習を構造的に抑制する」性質を持つため。

## PEFT 手法の分類

### 1. 加算型（Additive / Adapter-based）

凍結された事前学習モデルに**小さな学習可能モジュール**を追加挿入する。

- **AdaptFormer**（Chen et al., NeurIPS 2022）: Transformer ブロック内に小型 MLP（bottleneck）を並列挿入。ViT 向けに最適化
- **Adapter**（Houlsby et al., 2019）: 元祖。各 Transformer 層の後に MLP を追加
- **ConvPass**（Jie & Deng, 2022）: 畳み込みベースの軽量バイパス

### 2. 低ランク型（Low-Rank Decomposition）

重み更新 $\Delta W$ を低ランク行列の積 $BA$ で近似する。

- **LoRA**（Hu et al., ICLR 2022）: $W \leftarrow W + BA$ ここで $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}, r \ll d$。NLP 発祥だが CV でも標準。Transformer の attention 重み（Q, V）に適用するのが定石
- **Fact-TT**（Jie & Deng, 2023）: テンソル分解ベースの拡張

### 3. プロンプト型（Prompt-based）

入力側に**学習可能なベクトル**（プロンプトトークン）を追加し、それだけを最適化。

- **Visual Prompt Tuning (VPT)**（Jia et al., ECCV 2022）: ViT の各層（VPT-Deep）または入力層のみ（VPT-Shallow）に学習可能トークンを追加
- **Prompt Tuning**（Lester et al., 2021、NLP 起源）

### 4. 選択型（Selective）

既存パラメータの**特定部分のみ**を更新する。

- **BitFit**（Zaken et al., 2022）: bias 項のみ更新（超軽量、全パラメータの 0.1% 以下）
- **DiffPruning**: 重要なパラメータのみ特定して更新

### 5. プロジェクション型

事前学習特徴量に対して軽量な変換を学習する。

- **VQT (Visual Query Tuning)**: query 部分のみ学習

## 主要手法の特性比較

| 手法 | 追加パラメータ | 推論コスト | CV ViT 性能 | 特徴 |
|---|---|---|---|---|
| **LoRA** | 〜0.5% | 推論時にマージ可（増加なし） | 高 | 最も汎用的、業界標準 |
| **AdaptFormer** | 〜0.2% | わずかに増加 | 高 | ViT 向け最適化 |
| **VPT-Deep** | 〜0.1% | わずかに増加 | 中〜高 | 解釈可能性が高い |
| **BitFit** | 〜0.05% | 増加なし | 中 | 超軽量、デバッグ容易 |
| **Full FT** | 100% | 増加なし | 高（多データ）/ 低（少データ） | 比較基準 |
| **Linear Probe** | 〜0.01% | 増加なし | 中（タスク次第） | 比較基準 |

## SeSL × PEFT の出会い

[[sources/revisiting-ssl-foundation-models]]（NeurIPS 2025）が明らかにした 3 つの重要な点：

1. **VFM × Labeled-only PEFT** は VFM × SSL（FixMatch/FlexMatch/SoftMatch）に匹敵する性能を出す。ラベルなしデータの恩恵は VFM 時代では限定的
2. **異なる PEFT 手法は同じ VFM 上でも多様な予測**を生む（LoRA と AdaptFormer の予測上位 20% にはかなりの相違）
3. その多様性を利用したアンサンブル疑似ラベリング **[[entities/v-pet]]** が既存 SSL を凌駕

これは「PEFT が SeSL の中心ツールになる」というパラダイム転換の宣言と言える。

## CV 領域での発展経緯

| 年 | 出来事 |
|---|---|
| 2019 | Adapter（Houlsby et al.）—— NLP で原型登場 |
| 2021 | LoRA（Hu et al.）—— NLP で大規模言語モデル時代を切り拓く |
| 2022 | LoRA、AdaptFormer、VPT、BitFit が CV に持ち込まれる |
| 2023 | V-PETL Bench（Xin et al.）等で CV PEFT のベンチマーク化 |
| 2024-25 | PEFT × SSL/SeSL（FineSSL、V-PET）が登場 |

## 限界・課題

1. **タスク特異的な最適 PEFT が不明**: タスクごとに LoRA/AdaptFormer/VPT のどれが最良かは事前に分からない
2. **理論的理解の不足**: なぜ低ランク更新で十分なのか、なぜ異なる PEFT が異なる予測を生むかは経験的観察にとどまる
3. **タスク特異的ハイパラ**: ランク $r$（LoRA）、bottleneck 次元（AdaptFormer）等の調整が必要
4. **大規模事前学習に依存**: PEFT は「強力な事前学習バックボーン」が前提。スクラッチでは効かない

## 参考

- [[sources/revisiting-ssl-foundation-models]]: PEFT が SeSL の主役になる転換点を実証した研究（NeurIPS 2025）
- [[entities/v-pet]]: PEFT アンサンブル疑似ラベリングによる SeSL アルゴリズム
- [[concepts/foundation-model]]: PEFT の前提となる基盤モデル
- [[concepts/semi-supervised-learning]]: SeSL の全体像。PEFT が新たな設計軸として加わる
- [[concepts/vision-transformer]]: PEFT は ViT 向けに最適化されることが多い
- [[entities/clip]]、[[entities/dinov2]]: PEFT 適用の代表的バックボーン
