---
type: entity
entity_kind: model
aliases: [EVA-X, EVA-X-Ti, EVA-X-S, EVA-X-B]
tags: [medical-imaging, chest-x-ray, foundation-model, self-supervised-learning, masked-image-modeling, vision-transformer, eva-clip, mgca, dual-vit, hustvl, npj-digital-medicine]
related: [[concepts/foundation-model]], [[concepts/masked-image-modeling]], [[concepts/self-supervised-learning]], [[concepts/vision-transformer]], [[entities/ibot]], [[entities/mae]], [[entities/clip]], [[entities/i-synmed]], [[sources/eva-x]]
sources: [[sources/eva-x]]
updated: 2026-05-28
---

# EVA-X

## 概要

**EVA-X** は華中科技大学（HUST）の Jingfeng Yao・Xinggang Wang らが **npj Digital Medicine 2025** で発表した、胸部 X 線画像専用の自己教師あり基盤モデル。**EVA-02 の医療版**として、「学習可能 ViT + 凍結 CLIP トークナイザ」の dual ViT 構造で MIM を行う。Merged-520K（CXR14+CheXpert+MIMIC-CXR）で事前学習し、11 下流タスクで SOTA を達成。

- 論文: "EVA-X: a foundation model for general chest x-ray analysis with self-supervised learning"
- DOI: 10.1038/s41746-025-02032-z
- 公開: 2025-11-17（npj Digital Medicine, Nature 系列）
- 所属: 華中科技大学 Vision Lab（hustvl）等
- コード・重み: https://github.com/hustvl/EVA-X

---

## モデルファミリー

| モデル | パラメータ | FLOPs | CXR14 mAUC |
|---|---|---|---|
| **EVA-X-Ti** | 6M | 1.26 G | 82.4 |
| **EVA-X-S** | 22M | 〜4 G | 83.3 |
| **EVA-X-B** | 86M | 〜18 G | **83.5**（SOTA） |

EVA-X-Ti は MGCA-B（86M、13× FLOPs）を上回る効率性。

---

## アーキテクチャ：Dual ViT

```
入力画像 x（224×224）
  ↓ パッチ分割（16×16, n=196 patches）
  ↓ 線形射影 + 位置符号化（RoPE）+ [CLS] トークン
  ↓ シーケンス Z = {z_0, z_1, ..., z_196}
  ↓
  ├──── 学習可能 EVA-X ViT
  │       ├── 30% トークンをマスクトークンに置換
  │       ├── Sub-LN + SwiGLU + RoPE 強化 transformer ブロック
  │       └── 出力 Z'_e（マスク位置のみ抽出）
  │
  └──── 凍結 CLIP トークナイザ（マスクなし全シーケンス処理）
          ├── 選択肢1: EVA-CLIP ViT-B/L/G（自然画像）
          └── 選択肢2: MGCA-ViT-B（医療画像 CLIP）
          └── 出力 Z'_t（同じマスク位置のみ抽出）

損失: maximize Σ cos(Z'_e[i], Z'_t[i]) for i ∈ mask_list
```

---

## 主要ハイパーパラメータ

| パラメータ | 値 |
|---|---|
| パッチサイズ | 16 |
| 入力解像度 | 224×224（336 リサイズ後にランダムクロップ）|
| マスク比 r | **0.3**（控えめ、MAE の 0.75 より低い）|
| 損失 | 各マスク位置でコサイン類似度最大化 |
| 位置符号化 | **Rotary Position Embedding (RoPE)** |
| Normalization | **Sub-LN**（Foundation Transformers 由来）|
| FFN 活性化 | **SwiGLU** |
| トークナイザ | EVA-CLIP (B/L/G) or MGCA-ViT-B |
| トークナイザの学習 | **完全凍結** |
| 訓練データ | Merged-520K（ラベル不使用）|
| 前処理 | 正面 (AP/PA) ビューのみ、bilinear resize → ランダムクロップ |

---

## 訓練データ：Merged-520K

| データソース | サイズ | 備考 |
|---|---|---|
| Chest X-Ray14（NIH）| 112K | 14 病理ラベルあるが事前学習では不使用 |
| CheXpert（Stanford）| 224K | 14 病理ラベル不使用 |
| MIMIC-CXR（MIT）| 200K+ | レポート不使用 |
| **合計** | **520K+** | **すべて純粋にラベルなし画像として使用** |

下流タスクのテストセット画像は事前学習から除外。

---

## 主要な実験結果

### CXR14 マルチラベル分類（事前学習表現品質指標）

| 手法 | パラメータ | mAUC |
|---|---|---|
| ResNet50（教師あり）| 25M | 〜81 |
| DenseNet121（教師あり）| 8M | 〜81 |
| MoCov2 | 23M | 〜80 |
| MAE（natural）| 22M | 〜81 |
| Medical MAE | 22M | 〜82 |
| MGCA | 86M | 81.8 |
| SelfMedMAE | 86M | 81.5 |
| **EVA-X-Ti** | 6M | **82.4** |
| **EVA-X-S** | 22M | **83.3** |
| **EVA-X-B** | 86M | **83.5** |

### CheXpert（5 病理）

EVA-X-Ti / EVA-X-S が全モデル中で最良 mAUC。EVA-X-Ti は 6M で全先行手法を上回る。

### COVID-19 単一ラベル分類

| データセット | EVA-X mAUC | 標準偏差 |
|---|---|---|
| CovidX-CXR-3 | **99.8** | 0.03 |
| CovidX-CXR-4 | **99.4** | 0.03 |

参考: Medical MAE std 0.045、MGCA std 0.055、BioViL std 0.135 → EVA-X は **訓練の安定性で異次元**。

### ラベル効率（COVID-19）

| 訓練データ量 | EVA-X | 他手法 |
|---|---|---|
| 1% | **95%** 精度 | 大幅劣後 |
| 10% | 〜97% | 〜90% |
| 100% | 99%+ | 〜95% |

### セグメンテーション

| タスク | EVA-X Dice | 比較最良 |
|---|---|---|
| 肺 | **95.49%** | — |
| 肺炎 | **54.51** | MAE 53.16 |
| 気胸 | **60.27** | MGCA 59.00 |
| 結核 | **60.10** | MAE 59.1 |

### 解釈可能性（Grad-CAM、CXR14 病変位置特定）

| モデル | mAP |
|---|---|
| ViT + MAE（先行）| 3.61 |
| **ViT + EVA-X** | **8.94**（+5.33pt 改善）|

EVA-X は ViT-MAE の弱点（CAM 性能 < CNN）を克服。

### 実世界データ（中国 14 病院、10K X 線）

| モデル | 平均 AUC | 最大 AUC |
|---|---|---|
| BioViL | 〜0.82 | — |
| MGCA | 〜0.83 | — |
| MedKLIP | 〜0.84 | — |
| Medical MAE | 〜0.85 | — |
| **EVA-X-S** | **0.8645** | **0.8788** |

---

## iBOT との対比

| 観点 | [[entities/ibot]] | **EVA-X** |
|---|---|---|
| トークナイザ | online（self-evolving）| **frozen external CLIP** |
| 拡張する元 | DINO + MIM | MIM 単独 |
| マスク比 | 〜30% | 30% |
| 訓練の複雑さ | EMA teacher 管理 | トークナイザ凍結のみ |
| 主用途 | 一般 SSL | 医療 X 線専用 |
| データ規模 | LVD-142M 等 | Merged-520K |

EVA-X = 「**iBOT の online tokenizer を、強い凍結外部モデルに置き換えた版**」

---

## EVA 系統での位置

```
EVA (CVPR 2023, Fang et al.) ── MIM + CLIP tokenizer の元祖
   ↓
EVA-02 (2024, Fang et al.) ── RoPE/Sub-LN/SwiGLU で改良
   ↓
EVA-CLIP ── EVA-02 ベースの巨大 CLIP（ViT-G）
   ↓
EVA-X (npj Digital Medicine 2025) ── 医療 X 線版
```

---

## 限界

1. **胸部 X 線専用**: CT/MRI/皮膚等への汎化は別途検証必要
2. **データバイアス**: CXR14/CheXpert/MIMIC は特定人口統計に偏る
3. **トークナイザ依存**: 強い CLIP（EVA-CLIP-G 等）に依存。他領域（病理画像等）では適用困難
4. **大規模言語モデルとの統合は未実装**: CheXagent/XrayGPT 等との比較なし
5. **iBOT との直接比較なし**: online vs frozen tokenizer の優劣は実証されていない

---

## 関連ページ

- [[sources/eva-x]]: 詳細な論文要約
- [[translations/eva-x]]: 日本語全文翻訳
- [[entities/i-synmed]]: 対照される医療 X 線 SSL（DDPM 合成 + DINO）
- [[concepts/foundation-model]]: EVA-X はその医療版
- [[concepts/masked-image-modeling]]: EVA-X が採用する事前学習パラダイム
- [[concepts/self-supervised-learning]]: SSL の全体像
- [[concepts/vision-transformer]]: バックボーンアーキテクチャ
- [[entities/mae]]: MIM の元祖。EVA-X が比較し凌駕
- [[entities/ibot]]: 最も近い設計思想（online vs frozen tokenizer）
- [[entities/clip]]: トークナイザとして使われる EVA-CLIP の親概念
- [[concepts/online-tokenizer]]: iBOT 提案。EVA-X の凍結外部トークナイザと対比
- [[concepts/knowledge-distillation]]: 強い teacher を凍結利用する発想
- [[concepts/rotary-position-embeddings]]: RoPE。EVA-X が採用
