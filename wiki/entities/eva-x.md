---
type: entity
entity_kind: model
aliases: [EVA-X, EVA-X-Ti, EVA-X-S, EVA-X-B]
tags: [medical-imaging, chest-x-ray, foundation-model, self-supervised-learning, masked-image-modeling, vision-transformer, eva-clip, mgca, dual-vit, hustvl, npj-digital-medicine]
related: ["[[concepts/foundation-model]]", "[[concepts/masked-image-modeling]]", "[[concepts/self-supervised-learning]]", "[[concepts/vision-transformer]]", "[[entities/ibot]]", "[[entities/mae]]", "[[entities/clip]]", "[[entities/i-synmed]]", "[[entities/medgemma]]", "[[entities/biomedclip]]", "[[concepts/medical-foundation-model]]", "[[sources/eva-x]]"]
sources: ["[[sources/eva-x]]"]
updated: 2026-08-28
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

## 対比: 汎用基盤モデルのドメイン適応（MedGemma）

**EVA-X と [[entities/medgemma]] は医療基盤モデルの両極**にある。同じ「胸部 X 線を読む」でも:

| | EVA-X | MedGemma 4B |
|---|---|---|
| **規模** | **6M（Ti）〜 22M（S）** | **4,000M** |
| **対象** | 胸部 X 線**のみ** | 放射線・病理・皮膚・眼底 + テキスト |
| **作り方** | ゼロから MIM（凍結 CLIP トークナイザ） | Gemma 3 + SigLIP を**医療データ 2% / 10% で混ぜ直す** |
| **出力** | 表現（分類・検出のバックボーン） | **テキスト**（レポート生成・VQA・推論） |
| **強み** | 極小で SOTA（EVA-X-Ti 6M で CXR14 mAUC 82.4）、**エッジで動く** | 多モダリティ兼任、微調整の出発点、言語との統合 |

**直接比較した研究は本 wiki の範囲にない**が、選択の軸ははっきりしている。**1 モダリティ・1 タスクで最高精度が要りエッジで動かしたいなら特化 SSL、複数モダリティを横断しテキストと組み合わせたいなら汎用のドメイン適応**である。

同じトレードオフは [[entities/medsiglip]]（MedGemma の視覚エンコーダ、400M で 4 領域兼任）の linear probe 結果にも現れており、**最小の 64 枚では胸部 X 線専用の CXR Foundation に負け、512 枚以降で逆転する**。多領域兼任の代償が極少データ領域に出る。

- [[concepts/medical-foundation-model]] — 医療基盤モデルの 3 類型（本モデルは「単一モダリティ特化 SSL」型）
- [[entities/medgemma]] / [[entities/medsiglip]] — 対極にある「汎用ドメイン適応」型

## 反例: 多様性が特化を上回る場合（BiomedCLIP）

**単一モダリティ特化という戦略には、正面からの反例がある。** [[sources/biomedclip]] は RSNA 肺炎検出（胸部 X 線）において、**放射線科特化の BioViL を、ラベル付きデータのわずか 10% で完全教師ありの BioViL を上回る**形で破っている。

論文はこの結果の安易な解釈を自ら潰している。

> **BiomedCLIP の事前学習で用いられた放射線関連の画像は BioViL の事前学習で用いられた MIMIC-CXR より多くはなく**、画像–テキスト対はおそらくはるかにノイジーである。これは BiomedCLIP の RSNA における優れた性能が**より多くの放射線科特有の事前学習に由来する可能性を排除する**。

つまり「放射線データを多く見たから勝った」のではなく、**統計図表や顕微鏡画像を含む非放射線科の画像型を含む多様性そのものが、放射線科の性能を上げた**（**正の転移**）。

**ただし BiomedCLIP が万能なわけでもない。** 同じ論文のゼロショット分類では、病理特化の **PLIP が LC25000（肺）で BiomedCLIP を 16pt 上回る**。特化モデルは自分の土俵では依然として強い。

> **EVA-X の立ち位置はこれで揺らぐわけではない。** EVA-X は **6M〜22M** という桁違いに小さいモデルで CXR14 mAUC 82〜83 を出す点に価値があり、エッジ配備や低計算資源の場面での優位は変わらない。**「精度で勝つための特化」から「効率で勝つための特化」へと、特化の意義が移った**と読むのが妥当である。この整理は [[concepts/medical-foundation-model]] を参照。

- [[entities/biomedclip]] / [[sources/biomedclip]] — 多様性による正の転移を示した研究
- [[entities/pmc-15m]] — その訓練データ（30 の画像型、1,528 万対）
