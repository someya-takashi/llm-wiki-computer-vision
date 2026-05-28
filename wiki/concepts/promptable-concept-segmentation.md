---
type: concept
aliases: [Promptable Concept Segmentation, PCS, コンセプトプロンプトセグメンテーション]
tags: [task, segmentation, open-vocabulary, foundation-model, prompt-engineering, concept]
related: [[promptable-segmentation]], [[zero-shot-transfer]], [[foundation-model]], [[contrastive-learning]]
sources: [[sources/sam-3]]
updated: 2026-05-26
---

# Promptable Concept Segmentation（PCS, コンセプトプロンプトセグメンテーション）

## 一言で

**「短い名詞句や画像 exemplar をプロンプトとして与えると、その視覚コンセプトに合致する画像/動画中の *すべてのインスタンス* を検出・セグメント・追跡するタスク」**。SAM 3（[[sources/sam-3]] / [[entities/sam-3]]）が 2025 年に定義した、CV foundation model の新タスクパラダイム。

PVS（[[concepts/promptable-segmentation]]、SAM 1/2 のタスク）が「**1 プロンプト → 1 オブジェクト**」だったのに対し、PCS は「**1 プロンプト → コンセプトの全インスタンス**」を返す。

> **補足: なぜ「concept」と呼ぶか** — "instance"（個別オブジェクト）でも "category"（閉じたクラス集合）でもなく、「視覚的に grounding 可能な任意の意味カテゴリ」を指すから。"yellow school bus" のような修飾子付き名詞句、"striped cat" のような属性付きカテゴリ、画像 exemplar で示された未知カテゴリも含む。語彙は open-vocabulary で、Wikidata ベースの ontology から無数に作れる。

## PVS との対比

| | **PVS**（[[concepts/promptable-segmentation]], SAM 1/2） | **PCS**（SAM 3 新規） |
|---|---|---|
| **プロンプト** | 点、ボックス、粗マスク（1 インスタンス指定） | 名詞句、画像 exemplar、両方の組み合わせ（コンセプト指定） |
| **出力** | 1 オブジェクトのマスク | コンセプトに合う **全インスタンス** のマスク + 一意 ID |
| **動画** | 1 オブジェクトの masklet | 複数オブジェクトの masklet 群（個別 ID 保持） |
| **対話的洗練** | 別フレームで点追加 | exemplar の正/負追加、NP 更新 |
| **典型応用** | データアノテーション（個別物体）、対話的編集 | open-vocabulary 検出、自動アノテーション、ロボティクス |

両者は **互補的**: SAM 3 は両方を統一モデルでサポート。PCS で全インスタンスを取得 → 個別を PVS で洗練、という pipeline が標準利用パターン。

## なぜ PCS が必要だったか

CV 応用の多くは「**シーン中のすべての X**」を要求する：

- **データアノテーション**: 「画像中のすべての車にマスク」（個別を指定する PVS では遅すぎる）
- **ロボティクス**: 「テーブル上のすべての赤いボウル」
- **動画編集**: 「動画を通してすべての人物を追跡」
- **AR/VR**: 「視野中のすべてのテキスト」「すべての顔」
- **科学**: 「顕微鏡画像中のすべての細胞」

既存の open-vocabulary 検出器（**GLIP** [[entities/glip]] / **Grounding DINO** [[entities/grounding-dino]] / OWLv2 / LLMDet 等）は部分的にこれを解いていたが、SAM 3 論文の評価では SA-Co/Gold で cgF₁ 一桁台しか取れないレベル。SAM 3 は単一モデルで **大幅な精度向上 + 対話性 + 動画対応** を実現した。

**GLIP（[[sources/glip]]）は PCS の精神的祖先**（2021 年に「テキスト名詞句で全インスタンスを検出する」を提示）。**[[sources/grounding-dino|Grounding DINO]]（2023）はその DETR 系での発展形** で、**SAM 3 の直接の前駆**。SAM 3 は Grounding DINO に **マスク生成 + 対話性 + 動画 + presence head**（GLIP/Grounding DINO の「文脈なし存在判断」の弱点解決）を加えた発展形。

## タスクの形式定義

入力:
- **メディア**: 画像または短い動画（30 秒以下）
- **コンセプトプロンプト** のいずれか:
  - **テキスト**: 単純な名詞句（NP）= 修飾子 + 名詞。例: "red apple", "striped cat"
  - **画像 exemplar**: バウンディングボックス + ラベル（正/負）
  - **両方の組み合わせ**

出力:
- マッチするすべての **インスタンスマスク** + 一意の ID
- 動画では各オブジェクトの **masklet**（時空間マスク、ID 一貫性あり）
- セマンティックマスク（全画素にコンセプト所属の二値ラベル）も出せる

## プロンプトの種類と挙動

| プロンプト型 | 挙動 |
|---|---|
| **テキスト NP のみ** | 動画全フレームに global 適用、open-vocabulary |
| **画像 exemplar 正** | exemplar に似たインスタンスを全て検出 |
| **画像 exemplar 負** | exemplar に似たインスタンスを除外 |
| **NP + exemplar** | NP で候補を絞り、exemplar で洗練 |
| **PVS 風点追加** | 個別 masklet の境界を SAM 2 スタイルで洗練 |

> **補足: NP は画像/動画全体に適用、exemplar は frame 単位** — NP は global prompt（"全フレームで犬"）、exemplar は local refinement（"このフレームで見逃したこの犬"）という非対称な使い方が標準。

## 曖昧性の扱い

PCS は本質的に曖昧:

- **多義語**: "mouse" デバイス vs 動物
- **主観的記述子**: "cozy", "large"
- **漠然とした句**: "brand identity"（grounding 不可能かも）
- **境界の曖昧性**: "mirror" にフレームは含まれるか
- **遮蔽・ブラ**: オブジェクトの境界が不明瞭

SAM 3 の対処：
1. **3 アノテーター/テストペア**（SA-Co/Gold）+ **oracle 評価**
2. データパイプライン/ガイドラインで曖昧性を最小化
3. モデルに **ambiguity module** を含める（出力候補を複数）
4. 「すべてのプロンプトはカテゴリ定義で一貫していなければならない」というルール（"fish" を "tail" の exemplar で洗練することは想定外）

## 評価メトリクス（SAM 3 が提案）

- **cgF₁** (classification-gated F1) = `100 × pmF₁ × IL_MCC`
  - **pmF₁** = positive media-phrase pair（ground-truth ありペア）での micro F1（位置特定品質）
  - **IL_MCC** = image-level MCC（"NP が画像/動画にあるか" の二値分類精度）
- **信頼度 0.5 以上の予測のみ評価** → キャリブレーション強制
- 動画版: **pHOTA**（positive HOTA、open-vocabulary tracking 精度）

> **補足: AP ではなく cgF₁ を主要指標にした理由** — AP は信頼度閾値に不可知で「実用上どの閾値で使うか」が宙ぶらりんになる問題がある。cgF₁ は閾値固定で「**この閾値で使ったときの実用性能**」を直接測る設計。foundation model の実応用評価としてより誠実。

## モデルアーキテクチャの典型パターン

SAM 3 の設計に基づくと、PCS を解くモデルは：

1. **テキスト・画像 aligned backbone**（PE, CLIP, SigLIP 系）
2. **Concept-conditioned detector**（DETR ベース + fusion encoder）
3. **Presence head**（**認識と位置特定を分離する鍵**）
4. **Exemplar encoder**（画像プロンプトを処理）
5. **Tracker**（動画対応、SAM 2 風 memory）
6. **Detection ↔ Tracking matching**（identity 一貫性）

特に **(3) Presence head** は SAM 3 固有の重要発明：
- **proposal クエリ** は「位置特定」のみ担当: $p(q_i \text{ is a match} | \text{NP が存在})$
- **presence token** は「認識」のみ担当: $p(\text{NP が画像に存在})$
- 最終スコア = presence × proposal

これにより proposal クエリの「全画像コンテキスト依存」と「局所判断」の衝突を解消。

## PCS の限界

1. **長い指示表現は対応外**: 「青いシャツの男性の右側の帽子の人」のような複合表現は単純 NP で表現できない
2. **推論は別系統**: 「最大の物体」「光源を遮るもの」のような推論は MLLM が必要 → **SAM 3 Agent**（MLLM が SAM 3 をツールとして使う）パターンで対応
3. **動詞や関係の表現困難**: 「投げている人」のような動作は NP では捉えにくい
4. **ドメイン外用語への弱さ**: 訓練分布外の専門用語は性能低下、合成データ + AI verifier でドメイン適応が必要
5. **動画推論はオブジェクト数に線形**: 同時 5 オブジェクト程度が近実時間限界
6. **本質的に曖昧**: oracle 評価でも限界あり

## 関連タスクとの位置づけ

| タスク | 入力 | 出力 | 違い |
|---|---|---|---|
| **PVS** | 点/ボックス/マスク | 1 オブジェクト | インスタンス指定 |
| **PCS** | 名詞句/exemplar | 全インスタンス | コンセプト指定 |
| Open-vocabulary detection | テキスト | 全インスタンスのボックス | マスクなし、対話性なし |
| Open-vocabulary segmentation | テキスト | セマンティックマスク | インスタンス分離なし |
| Visual grounding | テキスト（長い） | 1 領域 | 指示表現で 1 つを指す |
| Referring segmentation | テキスト（長い） | 1 マスク | 指示表現で 1 つをセグメント |
| Video Instance Segmentation (VIS) | （訓練語彙固定） | 全インスタンス masklet | closed vocabulary |
| MOT | （訓練語彙固定） | tracking | open-vocab なし |

PCS は VIS + open-vocabulary + 対話性 + プロンプト型多様性を統合した、より一般的な定式化。

## 後続研究の予想

SAM 3 が 2025/11 出たばかりなので確実ではないが：

- **SAM 3 + 動画 VLM**: SAM 3 Agent パターンの拡張、長尺動画理解
- **SAM 3 + 3D**: 3D シーンの PCS への拡張（NeRF, Gaussian Splatting と統合）
- **SAM 3 + ロボット行動**: PCS 出力を行動方策の input に
- **PCS データの大規模公開**: SA-Co の追従、他組織の競合データ
- **AI verifier アプローチの一般化**: 他 CV タスクのアノテーション加速

## 関連ページ

- [[sources/sam-3]] — PCS を定義した SAM 3 原典の要約
- [[entities/sam-3]] — PCS を解く SAM 3 モデル
- [[entities/sa-co]] — PCS の訓練・評価データ
- [[concepts/promptable-segmentation]] — PVS（SAM 1/2 の元タスク、PCS と互補）
- [[concepts/zero-shot-transfer]] — PCS はゼロショット転移の最新形態
- [[concepts/foundation-model]] — PCS は CV foundation model の新軸
- [[entities/perception-encoder]] — SAM 3 が PCS のために採用する backbone
- [[entities/sam-2]] — SAM 3 が tracker として継承する元
