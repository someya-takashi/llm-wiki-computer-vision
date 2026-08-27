---
type: entity
entity_kind: model
aliases: [MedGemma 1.5, MedGemma 1.5 4B, MedGemma-1.5-4B]
tags: [medgemma, medgemma-1-5, medical-imaging, foundation-model, vision-language, 3d-imaging, whole-slide-imaging, long-context, localization, google, hai-def]
related: ["[[entities/medgemma]]", "[[entities/medsiglip]]", "[[entities/gemma-3]]", "[[entities/qwen3-vl]]", "[[concepts/medical-foundation-model]]", "[[concepts/object-detection]]", "[[concepts/foundation-model]]"]
sources: ["[[sources/medgemma-1-5]]"]
updated: 2026-08-27
---

# MedGemma 1.5

Google Research + Google DeepMind による [[entities/medgemma]] の後継（arXiv:2604.05081）。**4B のみ**（27B は 1.0 のまま据え置き）。詳細解説: [[sources/medgemma-1-5]] / 翻訳: [[translations/medgemma-1-5]]。

## 一言で

**アーキテクチャを一切変えずに** 3D CT/MRI・病理の全スライド画像（WSI）・解剖学的局在化・経時的 CXR 解析を単一の 4B モデルに載せた。効かせたのは**前処理と [[entities/gemma-3]] の 128K 長コンテキスト**である。

## MedGemma 1 4B との差分

| 項目 | MedGemma 1 4B | **MedGemma 1.5 4B** |
|---|---|---|
| **アーキテクチャ** | Gemma 3 4B + SigLIP-400M | **同一**（変更なし） |
| **視覚エンコーダ** | SigLIP を医療微調整 → MedSigLIP を作る | **MedSigLIP を凍結して使う** |
| **更新対象** | エンコーダ + 言語デコーダ | **言語デコーダのみ** |
| **蒸留の教師** | 大規模 IT 教師 1 つ | 改善された IT 教師 + **モダリティ別の補助教師**（CT / MRI / 病理） |
| **蒸留の方式** | — | トークンあたり **256 の教師ロジット**を教師確率で重み付けサンプル |
| **入力できるもの** | 2D 画像 + テキスト | **+ 3D ボリューム / WSI / 画像対（経時）** |
| **出力できるもの** | テキスト | **+ バウンディングボックス（JSON）** |

<figure>

![](../../raw/assets/medgemma-1-5/fig1.png)

<figcaption>図1（再掲）: コレクションの対応範囲。MedGemma 1.5 4B（緑の帯）だけが 2D imaging / Text / Advanced imaging の 3 領域すべてに届く。MedGemma 27B は 2D + Text、MedSigLIP は 2D imaging のみ。</figcaption>
</figure>

## 高次元入力の前処理仕様

**これが本モデルの実質的な中身である。**

### 3D ボリューム（CT / MRI）

| 項目 | 値 |
|---|---|
| 変換 | **軸位断の 2D スライス列**、各 896×896 に再スケール |
| 最大スライス数 | **85** |
| 視覚トークン数 | **21,760** |
| 制約の根拠 | プロンプト（適応）と出力（所見）を含めて**合計 32K トークン以内**に収める |
| 超過時 | **z 軸に沿って等間隔サンプリング** |
| 選択基準 | 512×512 以内 / 軸位断 / 同一スライス厚 / 5 スライス以上 |
| 積むもの | CT: 同一スキャンの異なる再構成カーネル / MRI: T1w・T2w・GRE・SWI |

**CT の HU → RGB ウィンドウ写像**（チャネルごとに理由がある）:

| チャネル | 範囲（HU） | 対象 | 理由 |
|---|---|---|---|
| **R** | (-1024, 1024) | 全体の形態 | 脳・胸部・腹部をまたぐため、肺実質から皮質骨まで境界を可視に保つ |
| **G** | (-135, 215) | 軟部組織 | **緑は輝度への寄与が最大**なので、テクスチャ感度を活かせる軟部を割り当て |
| **B** | (0, 80) | 脳実質・出血 | 灰白質/白質の区別、急性頭蓋内出血、血管の石灰化 |

**MRI はウィンドウ処理をしない。** ボクセル値が相対的で生理学的ウィンドウが存在しないため、**ボリュームごとの min-max 正規化**で R=G=B の同値にする。

### 病理 WSI

| 項目 | 値 |
|---|---|
| 組織マスク | **5x 倍率**で生成、**HSV 色空間**のカスタム多段階セグメンテーション |
| 倍率の選択 | **1 スライドにつき 1 つを確率的に選択**: P(5x)=0.34 / P(10x)=0.33 / P(20x)=0.33 |
| パッチ | 896×896 の**非重複**、格子状、ストライド = パッチサイズ |
| 最大パッチ数 | **126** |
| 視覚トークン数 | **32,256** |
| 順序 | **元の空間的順序を保存**（相対的な位置の文脈を保つため） |

CT の 85 と WSI の 126 で上限が違うのは、**伴うテキスト入力の長さの差**（CT は放射線レポートで長い、WSI はキャプションが短い）による予算配分。

## 主要結果

### 新しい能力（MedGemma 1 はほぼできなかった）

| Task | Metric | MedGemma 1 4B | **1.5 4B** | Qwen3 VL 4B | Gemini 3.0 Pro | External SOTA |
|---|---|---|---|---|---|---|
| WSI Histopath | ROUGE-L | **2.2** | **49.4** | ? | 12.2 | 49.8（PolyPath） |
| Chest ImaGenome（局在化） | Mean IoU | **3.1** | **38.0** | 8.7 | 39.1 | 30.7-34.4（CoCa-CXR） |
| MRI Dataset 1 (3D) | Accuracy | 51.3 | **64.7** | 49.6 | 55.5 | – |
| CT Dataset 1 (3D) | Accuracy | 58.2 | **61.1** | 52.8 | 61.0 | – |
| MS-CXR-T（経時） | Macro-Acc | 61.1 | **65.7** | 53.5 | 62.9 | 68.5（BioViL-T） |
| EHR Dataset 4（検査レポート） | Macro F1 | 25 | **64** | – | 81 | – |
| **CT-RATE (OOD, 3D)** | Macro F1 | 23.5 | **26.9** | – | **8.5**（3.0 Flash） | – |

**WSI の 2.2 と局在化の 3.1 はほぼランダム出力の水準**である。「+47pt の改善」ではなく **「まったくできなかったことができるようになった」** と読むのが正確で、しかも**一気に外部 SOTA と並ぶか上回る**（局在化は CoCa-CXR を超えた）。

**CT-RATE（分布外）では Gemini 3.0 Flash の 3 倍以上**。3D 医療画像ではモデル規模より前処理と訓練データの適合が支配的。

### 既存タスク

**改善**: EHRQA 67.6 → **89.6**（+22.0）、EyePACS 64.9 → **76.8**（+11.9）、MedXpertQA (MM) 18.8 → **26.4**、RadGraph F1 21.9 → **27.2**、MedQA 64.4 → **69.1**、MedMCQA 55.7 → **59.8**、AfriMed-QA 52.0 → **56.0**

**退行**: **SLAKE 72.3 → 59.8（-12.5）**、PubMedQA 73.4 → 67.6（-5.8）、**MMLU Pro 39.1 → 33.8**（Gemma 3 4B の 43.6 から **-9.8**）、VQA-RAD 49.9 → 48.1、CXR14 50.1 → 48.4

## Qwen3-VL 4B との対比

論文が「明確に異なる設計思想」と呼ぶ非対称。

| 領域 | MedGemma 1.5 4B | [[entities/qwen3-vl\|Qwen3 VL 4B]] |
|---|---|---|
| MedQA | 69.1 | **76.8** |
| MMLU Med | 69.7 | **78.3** |
| EHRNoteQA | 80.4 | **90.6** |
| MIMIC-CXR / CheXpert / PathMCQA / EyePACS | **すべて勝ち** | 負け |
| Chest ImaGenome (IoU) | **38.0** | 8.7 |
| 3D CT / MRI | **61.1 / 64.7** | 52.8 / 49.6 |

> **すべての視覚タスクにおいて MedGemma 1.5 は Qwen3 VL 4B を上回った。**

**テキストの医学知識は汎用が勝ち、医療画像の解釈は特化が勝つ。** 医学知識は Web にテキストとして大量にあるが、眼底写真や病理パッチのラベル付きデータはない、という説明が付く。

## 弱点

- **「本当の 3D」ではない**。軸位断スライスを最大 85 枚まで等間隔サンプルするので、**薄いスライスでしか見えない小病変は原理的に落ちうる**
- **4B クラスの明確なトレードオフ**: MMLU Pro が Gemma 3 4B 比 -9.8pt。論文自身が「画像への特化のための集中的な微調整が領域外の汎用タスクの能力を減じた可能性」と明記
- **SLAKE の -12.5pt** は大きい（論文はベンチマーク側の限界を指摘して反論）
- **代表図（図3）に幻覚が写っている**。肝腫瘍の例で放射線科医が「実質的な肝内胆管拡張は明らかでない」と不同意
- **推論効率で不利**: CT-RATE では 18 状態を **1 検査あたり 18 回問い合わせ**る必要があった。特化 CT アーキテクチャは単一の順伝播で多ラベルを出す
- **訓練データの中核が非公開**（CXR-IND1 60 万、CT Dataset 1 28 万、MRI Dataset 1 17 万、Internal WSI 34 万）
- **RadGraph F1 が 1.0 の論文と食い違う**（本論文の MedGemma 1 4B は 21.9、[[sources/medgemma]] では 29.5）。プロンプトの標準化のため。**世代をまたいだ数値比較には注意**
- **系統的な人間評価がない**。1.0 にあった 306 症例の放射線科医評価に相当するものがなく、新能力ほど自動指標だけで測られている

## 追加された訓練データ

**表**: MedGemma 1 に対する追加分（★ = 非公開の内部データ）

| モダリティ | データセット | 事例数 | 段階 |
|---|---|---|---|
| 放射線 | ★ CXR-IND1（インドの大規模病院） | 605,732 | PT, Distill, RL |
| | ★ CT Dataset 1（頭部・胸部・腹部） | 282,963 | PT, Distill, RL |
| | ★ MRI Dataset 1（頭部・腹部・膝） | 167,674 | PT, Distill, RL |
| | Chest ImaGenome | 39,968 | RL |
| 病理 | ★ Internal WSI Histopathology | 335,825 | PT, RL |
| 皮膚科 | ★ Dermatology Dataset 4（日本） | 25,560 | PT, Distill |
| | ★ Dermatology Dataset 5（ラベルなし） | 87,879 | PT, Distill, RL |
| | ISIC（CC-0） | 40,269 | PT, Distill |
| EHR | EHRQA（Synthea 合成 FHIR） | 9,809 | Distill |
| | ★ EHR Dataset 2 / 3 / 4 / 5 | 計 約 4.2 万 | Distill |

## 公開

- <https://goo.gle/medgemma> / HAI-DEF: <https://goo.gle/hai-def>
- **4B のみ**。27B は [[entities/medgemma]] の 1.0 版が引き続き利用可能

## 関連ページ

- [[sources/medgemma-1-5]] — 論文要約（最重要）
- [[translations/medgemma-1-5]] — 全文和訳（Appendix A 込み）
- [[entities/medgemma]] / [[sources/medgemma]] — 前身。アーキテクチャは共通
- [[entities/medsiglip]] — 1.5 では**凍結**して使われる視覚エンコーダ
- [[entities/gemma-3]] — 128K 長コンテキストの出どころ
- [[entities/qwen3-vl]] — 同サイズの汎用 MLLM。設計思想の対比
- [[concepts/medical-foundation-model]] — 医療基盤モデルの類型
- [[concepts/object-detection]] — バウンディングボックス出力による局在化の文脈
