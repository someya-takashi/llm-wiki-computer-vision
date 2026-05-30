---
type: entity
entity_kind: dataset
aliases: [MMPR, MMPR-v1, MMPR-v1.2, MultiModal PReference dataset]
tags: [dataset, preference-optimization, multimodal, cot, opengvlab]
related: [[entities/mpo]], [[entities/internvl-3]]
sources: [[sources/mpo]]
updated: 2026-05-29
---

# MMPR — MultiModal PReference dataset

## 概要

**MMPR** = **MultiModal PReference dataset**。OpenGVLab Shanghai AI Lab が [[sources/mpo|MPO]] 論文（2024 Nov）で公開した、**約 300 万サンプルのマルチモーダル選好データセット**。**MLLM の Chain-of-Thought 推論強化に特化** した初の大規模 PO データ。

- 提案論文: [[sources/mpo]]（arXiv:2411.10442）
- HuggingFace: `OpenGVLab/MMPR-v1.2`（InternVL 3 用拡張版）
- ライセンス: MIT
- 詳細: [[sources/mpo]] / [[entities/mpo]]

---

## サイズと内訳

| カテゴリ | サンプル数 |
|---|---|
| 正解あり指示（correctness-based pipeline） | **2.5M** |
| 正解なし指示（DropoutNTP pipeline） | **750K** |
| **合計** | **約 3M** |

### トークン統計

| カテゴリ | 指示平均 | chosen 平均 | rejected 平均 | chosen 最長/最短 | rejected 最長/最短 |
|---|---|---|---|---|---|
| 正解なし | 25.0 | 211.4 | 171.2 | 1342 / 20 | 1642 / 17 |
| 正解あり | 79.5 | 300.0 | 350.5 | 2018 / 32 | 4097 / 33 |

---

## 構築パイプライン（2 種）

### (A) 正解あり指示 → 正誤判定ベース（2.5M）

```
1. InternVL2-8B に CoT 形式で 32 解答をサンプリング（「Final Answer: ***」で締める）
2. 正解と一致 → chosen
3. 不一致 / 最終答えなし → rejected
4. 1 クエリあたり最大 15 ペア
5. Temperature 1.0
```

**対象ドメイン**: Science / Chart / Mathematics / OCR

### (B) 正解なし指示 → DropoutNTP（750K）

```
1. InternVL2-8B から応答 y を生成 → chosen
2. y を半分で切り詰める → y_<j (前半)
3. InternVL2-8B に y_<j のみを渡し、画像なしで残りを補完 → ~y_>=j
4. rejected = [y_<j, ~y_>=j]
```

**着想**: 画像なしの補完は **必ずハルシネーションを含む** ため、$y$ と $\tilde{y}$ の半順序関係が保証される。

**対象ドメイン**: General VQA / Document（正誤判定が困難なため DropoutNTP のみ）

### Dropout Ratio のアブレーション

| DR | Object HalBench Resp.(↓) | MMHal Hall.(↓) |
|---|---|---|
| 0.25 | 9.3 | 40.6 |
| **0.50** | **7.6** | **31.3** |
| 0.75 | 11.6 | 36.5 |

**半分（DR=0.5）が最適**。0.25 だと品質差大きすぎ、0.75 だと共通プレフィックスが多すぎ。

---

## データソース（18 データセット × 6 ドメイン）

| Task | Datasets |
|---|---|
| **General VQA** | VQAv2, GQA, OKVQA, IconQA |
| **Science** | AI2D, ScienceQA, M3CoT |
| **Chart** | ChartQA, DVQA, MapQA |
| **Mathematics** | GeoQA+, CLEVR-Math, Geometry3K, GEOS, GeomVerse, Geo170K |
| **OCR** | OCRVQA, InfoVQA, TextVQA, STVQA, SROIE |
| **Document** | DocVQA |

---

## 多モーダル CoT 手法（3 種、データサンプリング時に併用）

1. **Background Knowledge-based CoT** （Science）: 関連背景知識 → 推論 → 最終答え
2. **Visual Content-based CoT** （Chart / OCR / Document）: 視覚内容分析 → 推論 → 最終答え
3. **Grounded CoT** （General VQA）: 応答内のオブジェクトを画像領域にリンク

---

## 効率性（vs RLAIF-V）

| 手法 | 選好ペアあたりトークン数 | 相対コスト |
|---|---|---|
| RLAIF-V (divide-and-conquer) | 992.7 | 100% |
| **DropoutNTP (ours)** | **571.2** | **57.5%** |

**性能ほぼ同等**:
- Object HalBench Resp.: RLAIF-V 7.3 vs **DropoutNTP 7.6**
- MMHal-Bench Score: RLAIF-V 3.5 vs **DropoutNTP 3.6**

**RLAIF-V の 57.5% コストで同等品質**、人手アノテーション不要、自動構築可能。

---

## 主な用途

### 1. InternVL2-8B-MPO の訓練（本論文）

[[entities/mpo|MPO]] アルゴリズムで InternVL2-8B を訓練、**MathVista +8.7、M3CoT +19.9**。

### 2. InternVL 3 の Stage 3 訓練（[[entities/internvl-3]]）

**MMPR v1.2（拡張版）** から **300K 選好ペア** を抽出して [[entities/internvl-3|InternVL 3]] の MPO 段階に使用。
- InternVL3-78B: 推論 Overall +4.1 ポイント
- InternVL3-38B: +4.5 ポイント

### 3. VisualPRM 訓練（VisualPRM400K）

[[entities/internvl-3|InternVL 3]] の **Process Reward Model（VisualPRM-8B）** の訓練データ **VisualPRM400K も MMPR v1.2 ベース**。

---

## MMPR-v1 と MMPR-v1.2 の違い

| 項目 | MMPR-v1 (本論文) | MMPR-v1.2 (InternVL 3) |
|---|---|---|
| サイズ | 3M ペア | 拡張版 |
| 用途 | InternVL2-8B-MPO | InternVL 3 SFT 後の MPO |
| ドメイン | 6 ドメイン × 18 データ | + 拡張データ |

---

## データスケーリング効果

10K → 100K で **Direct/CoT 両方で性能向上**:

| Data Size | Direct | CoT |
|---|---|---|
| 10K | 65 | 67 |
| 40K | 70 | 73 |
| 70K | 73 | 76 |
| **100K** | **76.4** | **78.9** |

**「選好データは多いほど CoT 性能が向上」**。

---

## 関連ページ

- 提案論文: [[sources/mpo]]
- 主な用途: [[entities/mpo]]（MPO アルゴリズム）/ [[entities/internvl-3]]（InternVL 3 Stage 3）
- 翻訳: [[translations/mpo]]
