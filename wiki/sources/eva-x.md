---
type: source
source_path: raw/papers/EVA-X_ a foundation model for general chest x-ray analysis with self-supervised learning - npj Digital Medicine.md
source_kind: paper
title: "EVA-X: a foundation model for general chest x-ray analysis with self-supervised learning"
authors: [Jingfeng Yao, Xinggang Wang, Yuehao Song, Huangxuan Zhao, Jun Ma, Yajie Chen, Wenyu Liu, Bo Wang]
year: 2025
venue: npj Digital Medicine
ingested: 2026-05-28
tags: [medical-imaging, chest-x-ray, foundation-model, self-supervised-learning, masked-image-modeling, vision-transformer, eva-clip, mgca, dual-vit, knowledge-distillation, hustvl]
translation: "[[translations/eva-x]]"
---

# EVA-X: a foundation model for general chest x-ray analysis with self-supervised learning

> 原典: [[translations/eva-x]] ・ `raw/papers/EVA-X_ a foundation model for general chest x-ray analysis with self-supervised learning - npj Digital Medicine.md`
> 著者: Jingfeng Yao, Xinggang Wang, Yuehao Song, Huangxuan Zhao, Jun Ma, Yajie Chen, Wenyu Liu, Bo Wang
> 所属: 華中科技大学（HUST、Vision Lab）等
> 出典: npj Digital Medicine 2025（Nature 系列、医療 AI 最高峰の一つ、2025-11-17 公開）
> コード: https://github.com/hustvl/EVA-X

## 一言まとめ

**胸部 X 線画像専用の自己教師あり基盤モデル**。EVA-02（自然画像基盤モデル）の設計思想を医療領域に展開し、「学習可能 ViT + 凍結 CLIP トークナイザ」という dual ViT 構造で MIM を行う。ラベルなし 520K 枚（CXR14+CheXpert+MIMIC-CXR）で事前学習し、11 個の下流 X 線タスク（分類・セグメンテーション・解釈可能性）で SOTA を達成。特筆すべきは **EVA-X-Ti（6M）が 13 倍の FLOPs を持つ MGCA-B を上回る**極端な効率性と、**1% ラベルだけで COVID-19 精度 95%** というラベル効率。

## 背景と問題意識

医療 AI の本質的問題：

- 胸部 X 線は世界の画像化処置の **40%**（年間 36 億件）を占めるが、深層学習モデルは：
  - **ラベル不足とラベル品質のばらつき**で汎化性能が弱い
  - **タスク特化的**で多様な臨床課題に対応しづらい
- AI 基盤モデルは NLP（GPT 等）と一般 CV（CLIP, DINOv2 等）で成功しているが、**胸部 X 線専用の汎用基盤モデルはまだ存在しない**

論文の問い：「**胸部 X 線版の DINOv2 / EVA-02 を作れないか？**」「**ラベルなしデータの大規模事前学習で、すべての胸部疾患タスクに汎化する単一モデルを構築できないか？**」

## 提案手法：EVA-X の核心設計

### Dual ViT アーキテクチャ

EVA-X の核は「**学習可能な ViT + 凍結された CLIP トークナイザ**」という 2 つの ViT による構造：

```
画像 x
  ↓ パッチ分割（16×16）+ 位置符号化 + [CLS] トークン
  ↓ シーケンス Z = {z_0, z_1, ..., z_n}
  ↓
  ├──── 学習可能 EVA-X ViT（mask token で 30% マスク） ─→ Z'_e
  │        ↑ RoPE + Sub-LN + SwiGLU（EVA-02 由来）
  │
  └──── 凍結 トークナイザ ViT（マスクなし） ─→ Z'_t
            ↑ EVA-CLIP（ViT-B/L/G）または MGCA-ViT-B（医療 CLIP）

損失: マスク位置でのコサイン類似度最大化
      maximize Σ cos(Z'_e[i], Z'_t[i]) for i ∈ mask_list
```

### 重要な設計判断

1. **マスク比 r=0.3（控えめ）**: MAE の 75% より遥かに低い。トークナイザが教師信号として強力なため、軽いマスクで十分という判断
2. **コサイン類似度損失**: MSE/L2 ではなく方向類似度。BEiT/iBOT の系統
3. **トークナイザは完全凍結**: iBOT の online tokenizer と対照的（後述）
4. **自然画像 CLIP も医療 CLIP も両方検証**: 興味深いことに、自然画像 EVA-CLIP も医療画像 X 線で有効。クロスドメイン転移の証拠
5. **2 つの ViT は同サイズか異サイズ**: トークナイザは「より大規模」を推奨（より豊富な特徴を提供）
6. **EVA-02 由来の改良**: RoPE / Sub-LN / SwiGLU / SubLN は ViT の安定性と性能を向上

### iBOT との対比

EVA-X は **iBOT の発想に近い**が、決定的に異なる：

| 観点 | [[entities/ibot]] | **EVA-X** |
|---|---|---|
| トークナイザの由来 | **online**（teacher が自己進化） | **external + 凍結**（EVA-CLIP/MGCA） |
| 拡張する元 | DINO（自己蒸留） | MIM 単独（自己蒸留なし） |
| マスク比 | 〜30%（同程度） | 30%（同じ）|
| 訓練の複雑さ | EMA teacher 管理が必要 | トークナイザ凍結のみ、極めて単純 |
| 訓練データ規模 | LVD-142M（DINOv2）等 | Merged-520K（医療特化）|

EVA-X は「**iBOT の online tokenizer を、強力な事前学習済み外部モデルに置き換えた版**」と位置づけられる。これは [[concepts/knowledge-distillation]] の発想——「**強い teacher を凍結して使う**」——の自然な応用。

### Merged-520K データセット

| 元データセット | サイズ | 役割 |
|---|---|---|
| Chest X-Ray14（NIH）| 112K | 主要訓練、CXR14 テストセットは除外 |
| CheXpert（Stanford）| 224K | 訓練のみ、200 枚のテストセットは除外 |
| MIMIC-CXR（MIT）| 200K+ | 訓練のみ |
| **合計** | **520K+** | **すべてラベルなしで使用** |

前処理: 正面ビュー（AP/PA）のみ、336×336 にリサイズ後 224×224 ランダムクロップ。

## 衝撃的な実験結果

### CXR14 線形プローブ性能（事前学習表現の品質）

| モデル | パラメータ | FLOPs | mAUC |
|---|---|---|---|
| DenseNet121（教師あり）| 8M | 〜3 G | 〜81 |
| ResNet50（教師あり）| 25M | 〜4 G | 〜81 |
| MoCov2 | 23M | 〜4 G | 〜80 |
| MAE（natural）| 22M | 〜4 G | 〜81 |
| Medical MAE | 22M | 〜4 G | 〜82 |
| MGCA | 86M | 〜18 G | 81.8 |
| SelfMedMAE | 86M | 〜18 G | 81.5 |
| **EVA-X-Ti（提案）** | **6M** | **1.26 G** | **82.4** |
| **EVA-X-S（提案）** | **22M** | **〜4 G** | **83.3** |
| **EVA-X-B（提案）** | **86M** | **〜18 G** | **83.5** ← SOTA |

**EVA-X-Ti（6M）が MGCA-B（86M、13× FLOPs）を上回る** という結果は、設計の優位性を端的に示す。

### 下流タスク 11 個すべてで SOTA

#### マルチラベル分類

| データセット | 比較最良 | EVA-X-Ti | EVA-X-S |
|---|---|---|---|
| CXR14（mAUC） | Kim 82.2 | 82.4 | **83.3** |
| CheXpert（mAUC） | Xiao 0.823 | **新 SOTA** | **新 SOTA** |

#### 単一ラベル分類（COVID-19）

| データセット | 最良比較 | EVA-X |
|---|---|---|
| COVID-CXR-3（mAUC） | 〜99 | **99.8**（std 0.03） |
| COVID-CXR-4（mAUC） | 〜99 | **99.4**（std 0.03） |

**標準偏差 0.03** は Medical MAE（0.045）、MGCA（0.055）、BioViL（0.135）より圧倒的に低い → 訓練の安定性が異次元。

#### ラベル効率（COVID-19）

**1% の訓練データで 95% の診断精度**を達成。これは他手法（数十パーセント程度の精度）に対して劇的な優位性。

#### セグメンテーション

| タスク | 比較最良 | EVA-X |
|---|---|---|
| 肺（Dice） | — | **95.49%** |
| 肺炎（Dice/Jaccard） | MAE 53.16/36.20 | **54.51/37.47** |
| 気胸（Dice/Jaccard） | MGCA 59.00/41.84 | **60.27/43.13** |
| 結核（Dice/Jaccard） | MAE 59.1/41.96 | **60.10/42.96** |

#### 解釈可能性（Grad-CAM、CXR14 病変位置特定）

EVA-X は ViT-MAE の弱点「CAM 性能が CNN より弱い」を克服。**ViT mAP を 3.61 → 8.94 に改善**。小病変位置特定で CNN を上回る。

#### 実世界データ（中国 14 病院 10K 枚）

EVA-X-S: 平均 AUC **0.8645**、最大 **0.8788**。Medical MAE / MGCA / BioViL / MedKLIP を上回る。Deepseek-v3 でレポートをアノテーション（F1 99% 対医師）。

## 既存 wiki との関係

### [[sources/i-synmed]]（IEEE Access 2025）との対比

両者とも「胸部 X 線 + SSL」だが、アプローチが全く異なる：

| 観点 | I-SynMed | **EVA-X** |
|---|---|---|
| ジャーナルの質 | IEEE Access（中堅） | **npj Digital Medicine（最高峰）** |
| データ規模 | 100K（合成）| **520K（実画像）** |
| データ性質 | DDPM 合成 | NIH+CheXpert+MIMIC 実 |
| SSL アルゴリズム | DINO（自己蒸留）| **MIM + CLIP トークナイザ** |
| バックボーン | ViT-16（DINO 標準）| EVA-02 改良 ViT |
| 主張 | 「合成 = 実」 | 「医療基盤モデルが作れる」|
| 規模感 | 1 GPU 500 epoch | 大規模事前学習 + 多タスク |

両者は「**胸部 X 線 SSL の二つの異なる道**」を代表する。I-SynMed が「合成データの可能性」を実証する**応用研究**であるのに対し、EVA-X は「**基盤モデルの本格構築**」を目指す**システム研究**。

### EVA-02 系統への系譜的位置

EVA-X は **EVA / EVA-02（Fang et al., 2023/2024）の医療版**：

```
EVA (CVPR 2023) ─→ MIM + CLIP tokenizer の元祖
   ↓
EVA-02 (2024) ─→ RoPE/Sub-LN/SwiGLU で安定性向上
   ↓
EVA-CLIP ────→ EVA-02 ベースの巨大 CLIP（ViT-G）
   ↓
EVA-X (npj Digital Medicine 2025) ─→ 医療 X 線版（EVA-CLIP をトークナイザに）
```

EVA 系統の特徴：「**CLIP 視覚エンコーダを凍結トークナイザとして MIM**」というレシピ。これは iBOT の online tokenizer を「**外部固定された強い教師**」に置き換えた発想。

## 限界・批判的視点

### 1. ドメイン特化の制約

論文自身が認める：「**胸部 X 線データのみで訓練しているため、他の医療タスク（CT、MRI、皮膚画像等）への性能は改善の余地がある**」。実用上、X 線専用モデルとして使う必要がある。

### 2. データバイアス

訓練データ（CXR14、CheXpert、MIMIC-CXR）は**特定の患者人口統計と疾病有病率に偏り**がある。Glocker et al. (2023) が指摘した「胸部 X 線基盤モデルのバイアスリスク」は EVA-X にも当てはまる。

### 3. トークナイザ選択の依存性

EVA-X の性能は**トークナイザの品質に強く依存**する。EVA-CLIP-G（巨大）を使えば高性能だが、別のトークナイザでは性能が落ちる可能性。「**強い CLIP がない医療領域**」では適用が困難。

### 4. オンライントークナイザとの比較が不十分

iBOT の online tokenizer 方式と直接比較していない。「**なぜ凍結外部 CLIP が良いか**」の理論的・実証的解明は今後の課題。

### 5. 評価データの一部リーク懸念

訓練データ（CXR14+CheXpert+MIMIC）に含まれる画像のテストセットは除外しているが、**同じ患者の別画像**が混入する可能性は完全には排除されていない。

### 6. 大規模言語モデルとの統合は未実装

Discussion で「**大規模医療言語モデルと組み合わせた視覚エンコーダとして機能可能**」と述べるが、本論文では実装されていない。CheXagent / XrayGPT 等との比較は将来課題。

## 研究上の位置づけ

### Wiki 内での位置

- **医療基盤モデルの代表例**: [[concepts/foundation-model]] の医療領域での実証
- **MIM の医療応用の到達点**: [[concepts/masked-image-modeling]] / [[entities/mae]] / [[entities/ibot]] の系統が医療に到達
- **EVA 系統の wiki 入口**: EVA / EVA-02 / EVA-CLIP の医療版を通じて、wiki に EVA ファミリーを導入
- **SSL の医療実用化**: [[concepts/self-supervised-learning]] の高度な応用事例
- **[[sources/i-synmed]] との対照ペア**: 同じ医療 X 線 SSL でも、設計思想とジャーナルの質が全く異なる 2 例の対比

### より広い研究文脈

EVA-X は以下の流れの帰着点：

```
[基盤モデル理論]
Foundation Models (Bommasani et al. 2021) — 概念提唱
   ↓
[一般 CV 基盤モデル]
CLIP (2021) → DINOv2 (2023) → EVA-02 (2024)
   ↓
[医療基盤モデル黎明期]
SAM (一般) → MedSAM (Ma et al. 2024) → EVA-X (2025)
   ↓
[未来: マルチモーダル統合]
EVA-X + LLM = 医療画像エージェント
```

「**汎用 CV 基盤モデルの設計レシピ（EVA-02 系統）が医療に正常に転移できる**」を示した重要な実証研究。

## 用語と略称

- **EVA-X** = 提案手法名（EVA の X-ray 版を意味すると推定）
- **EVA-02** = Fang et al. 2024 の自然画像基盤モデル。EVA-X の設計の母体
- **EVA-CLIP** = EVA-02 ベースの巨大 CLIP モデル（ViT-G まで）。EVA-X のトークナイザとして使用可
- **MGCA** = Multi-Granularity Cross-modal Alignment（Wang et al., NeurIPS 2022）。医療 CLIP の代表
- **MGCA-ViT-B** = MGCA で訓練された ViT-B 視覚エンコーダ。EVA-X の医療トークナイザ
- **Merged-520K** = EVA-X 訓練用に CXR14+CheXpert+MIMIC-CXR を統合した 520,000 枚データセット
- **CXR14 / Chest X-Ray14** = NIH 公開の胸部 X 線データセット（112K、14 病理）
- **CheXpert** = Stanford 公開の胸部 X 線データセット（224K、14 病理）
- **MIMIC-CXR** = MIT 公開の胸部 X 線データセット（200K+、レポート付き）
- **COVID-CXR-3 / COVID-CXR-4** = COVID-19 検出データセット
- **RSNA Pneumonia** = 肺炎検出データセット
- **SIIM-ACR Pneumothorax** = 気胸セグメンテーションデータセット
- **Tuberculosis (Shenzhen)** = 結核データセット
- **Lung Segmentation (YoushanZhang)** = 肺セグメンテーション GitHub データセット
- **dual ViT** = 学習可能 ViT + 凍結トークナイザ ViT の 2 つの構造（EVA-X 設計）
- **MIM** = Masked Image Modeling（マスク画像モデリング）。詳細: [[concepts/masked-image-modeling]]
- **mask token** = マスクされた位置に挿入する学習可能ベクトル
- **mask ratio r** = マスクするトークンの割合。EVA-X では 0.3
- **mask_list** = マスクされた image token のインデックス集合
- **RoPE** = Rotational Positional Encoding。詳細: [[concepts/rotary-position-embeddings]]
- **Sub-LN** = Sub-Layer Normalization（Foundation Transformers より）
- **SwiGLU** = Swish-Gated Linear Unit（Shazeer 2020）
- **MHSA** = Multi-Head Self-Attention
- **FFN** = Feed Forward Network
- **CLS token** = 分類用の特別トークン
- **CAM** = Class Activation Map
- **Grad-CAM** = 勾配ベースの CAM（Selvaraju et al. 2017）
- **UperNet** = セグメンテーションヘッドアーキテクチャ
- **UNet** = encoder-decoder セグメンテーションアーキテクチャ
- **mAUC** = mean Area Under the Curve（複数クラス平均 AUC）
- **mAcc** = mean Accuracy
- **TPR / FPR** = True/False Positive Ratio
- **Dice / Jaccard** = セグメンテーション精度指標
- **AP** = Average Precision
- **IoU** = Intersection over Union
- **PA/AP view** = Posterior-Anterior / Anterior-Posterior（X 線の正面ビュー）
- **Ark+** = 完全公開された医療 AI 基盤モデル（Ma et al., Nature 2025）
- **CXR-Foundation** = Google の X 線基盤モデル（ELIXR、Xu et al. 2023）
- **CheXagent** = LLM ベースの胸部 X 線解釈モデル（Chen et al. 2024）
- **XrayGPT** = X 線特化大規模視覚言語モデル
- **Deepseek-v3** = 中国の大規模言語モデル。本研究では実世界データのレポート解析に使用
- **F1 score** = 精度と再現率の調和平均
- **bilinear 補間** = 画像リサイズの標準手法
- **MoCo / MoCov2** = Momentum Contrast（対比 SSL）
- **Medical MAE** = MAE の医療版
- **SelfMedMAE** = 別の医療 MAE バリアント
- **BioViL** = Microsoft の医療視覚言語モデル
- **MedKLIP** = 医療知識強化 CLIP
- **GLoRIA** = 医療マルチモーダル局所表現学習
- **DenseNet121** = 密結合畳み込みネット
- **DeiT** = データ効率 ViT
- **CI** = Confidence Interval（信頼区間）

## 関連ページ

- [[translations/eva-x]]: 日本語全文翻訳
- [[entities/eva-x]]: EVA-X モデルのスペックシート
- [[sources/i-synmed]]: 対照される医療 X 線 SSL 論文（合成データ + DINO）
- [[concepts/foundation-model]]: 基盤モデル（EVA-X はその医療版）
- [[concepts/self-supervised-learning]]: SSL の全体像
- [[concepts/masked-image-modeling]]: EVA-X が採用する事前学習パラダイム
- [[concepts/vision-transformer]]: バックボーンアーキテクチャ
- [[entities/mae]]: MIM の元祖。EVA-X が比較し凌駕
- [[entities/ibot]]: 最も近い設計思想（online tokenizer vs 凍結外部 tokenizer）
- [[entities/clip]]: トークナイザとして利用される EVA-CLIP の親概念
- [[concepts/online-tokenizer]]: iBOT 提案。EVA-X の凍結外部トークナイザと対比
- [[concepts/knowledge-distillation]]: 強い teacher を凍結利用するという発想
- [[concepts/rotary-position-embeddings]]: RoPE。EVA-X が採用
