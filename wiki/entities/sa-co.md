---
type: entity
entity_kind: dataset
aliases: [SA-Co, Segment Anything with Concepts, SACo]
tags: [dataset, benchmark, segmentation, concept, open-vocabulary, meta-superintelligence-labs, hard-negatives, ai-verifier]
related: [[entities/sam-3]], [[concepts/promptable-concept-segmentation]], [[entities/sa-v]], [[entities/sa-1b]]
sources: [[sources/sam-3]]
updated: 2026-05-26
---

# SA-Co（Segment Anything with Concepts）

## 概要

**SA-Co** = **Segment Anything with Concepts**。Meta Superintelligence Labs が SAM 3（[[entities/sam-3]]）のために構築・公開する、**コンセプトベースのオープン語彙セグメンテーションのための訓練・評価データセット群**。**4M unique noun phrases** と **207K concepts のベンチマーク**（既存の 50 倍以上）を含み、open-vocabulary 視覚 segmentation の de facto 標準を目指す。

- 公開: SAM 3 公開と同時（モデルと共に Apache 2.0/CC by 4.0 想定）
- 詳細: [[sources/sam-3]] §4-5
- 公式: SAM 3 公開時に詳細予定

> **補足: なぜ「with Concepts」か** — SA-1B（画像 + マスクのみ、ラベルなし）、SA-V（動画 + masklet、ラベルなし）と異なり、SA-Co は **画像/動画 + マスク + コンセプトラベル（名詞句）** の 3 つ組を持つ。コンセプトラベルこそが PCS タスクの本質的入力なので、データセット名自体に "Concepts" を冠している。

## 構成: 5 つのデータグループ

| 部分 | 用途 | サイズ | 取得方法 |
|---|---|---|---|
| **SA-Co/HQ** | 訓練（高品質） | **5.2M 画像 + 4M unique NP + 52M マスク** | データエンジン 4 phase（人間 + AI verifier） |
| **SA-Co/SYN** | 訓練（合成） | **38M 句 + 1.4B マスク** | 成熟データエンジン（Phase 3）で人間関与なし |
| **SA-Co/EXT** | 訓練（外部） | 15 既存データセット | 外部データに hard negative を強化 |
| **SA-Co/VIDEO** | 訓練（動画） | **52.5K 動画 + 24.8K unique NP + 134K video-NP** | データエンジン Phase 4 |
| **SA-Co Benchmark** | 評価 | **207K concepts + 121K 画像/動画 + 3M+ メディア-句ペア** | 別途 4 分割（Gold/Silver/Bronze/Bio/VEval） |

## SA-Co/HQ（高品質訓練データ）

### 規模と特徴

- **5.2M 画像** に **4M unique noun phrases** をアノテート
- **52M インスタンスマスク**
- 「最大の高品質 open-vocab セグメンテーションデータセット」と論文で主張
- **15 ドメイン** をカバー（Phase 3 で拡張）

### データエンジン 4 phase

| Phase | モデル・イン・ザ・ループ | 主要新規 | 収集量 |
|---|---|---|---|
| **1: Human Verification** | SAM 2 + 検出器でマスク提案 → 人間が MV (Mask Verification) と EV (Exhaustivity Verification) | ベースライン、初期 SAM 3 訓練データ | 4.3M image-NP |
| **2: Human + AI Verification** | Llama 3.2 を MV/EV にファインチューン → **AI verifier 化**、NP 生成も Llama で **hard negative 含む** | スループット 2 倍化 | +122M image-NP |
| **3: Scaling + Domain Expansion** | **SA-Co ontology**（Wikidata ベース、22.4M ノード、17 トップカテゴリ、72 サブカテゴリ）から long-tail NP マイニング | ドメイン 15 個に拡大、SAM 3 を 7 回・AI verifier を 3 回更新 | +19.5M image-NP |
| **4: Video Annotation** | 成熟画像 SAM 3 を動画拡張、シーン/動き filter、混雑・追跡失敗シーン優先 | 動画への展開 | 52.5K 動画 / 467K masklet |

### Hard Negatives（鍵となる工夫）

通常のデータでは「正しい NP」のみが付与されるが、SA-Co は **AI 検証者を欺くような hard negative NP** も自動生成して訓練に含める：

- 例: 画像に "cat" がいるとき、"dog"（明らかな negative）ではなく "tiger" や "leopard"（似ているが違う）を hard negative として
- これにより SAM 3 の **image-level recognition (IL_MCC)** が劇的改善（0.44 → 0.68、アブレーション結果）

### AI Verifier

**Llama 3.2** を MV (Mask Verification) と EV (Exhaustivity Verification) タスクにファインチューン：

- MV: 与えられたマスクが NP に合致する品質か（多肢選択評価）
- EV: 画像中の NP のすべてのインスタンスがマスクされているか

AI verifier は **人間に近い精度** で、人間の努力を最も困難なケースに集中させる。これにより：

- スループット 2 倍化
- 新ドメインへの zero-shot 適応も可能（MV は zero-shot で機能、EV は控えめな domain-specific 人間教師あり学習で改善）

## SA-Co/SYN（合成訓練データ）

- **38M 句 + 1.4B マスク**
- 成熟データエンジン（Phase 3）で **人間関与なし** に生成
- SAM 3 + AI verifier だけでスケール可能

ドメイン適応実験（図 9）で **SYN が HQ（人間あり）と類似のスケーリング** を示すことが示された：

> **補足: SYN ≈ HQ の意味** — これは「**人間アノテーションコストなしで新ドメインへ展開できる**」というのが foundation model 時代の重要な発見。Phase 3 までで成熟した SAM 3 + AI verifier の系を、未訓練ドメインに適用すれば人間 0 でデータが取れる。データセット拡張のスケーラビリティが根本的に変わる。

## SA-Co/EXT（外部データ）

- **15 個の既存データセット**
- インスタンスマスクアノテーション付き
- **SA-Co ontology を使って hard negative を強化**

外部データを取り込みつつ、SAM 3 の hard negative 訓練の方針と整合させる仕組み。

## SA-Co/VIDEO（動画訓練データ）

- **52.5K 動画 + 24.8K unique NP + 134K video-NP pair**
- 動画平均 84.1 フレーム / **6 fps** アノテーション
- 467K masklet（SAM 3 動画拡張版で生成、人間で検証）

## SA-Co Benchmark（評価データ）

PCS タスクのオープン語彙評価のための新ベンチマーク：

| 項目 | 値 |
|---|---|
| ユニークコンセプト | **207K**（既存ベンチマークの **50 倍以上**） |
| 画像 + 動画 | 121K |
| メディア-句ペア | 3M+（hard negative ラベル含む） |
| 分割数 | 4 |

### 4 分割

| 分割 | 内容 | ドメイン数 | アノテーター数/ペア | 用途 |
|---|---|---|---|---|
| **SA-Co/Gold** | 高品質、3 人検証 | 7 | **3**（曖昧性のため） | 人間性能測定、最重要評価 |
| **SA-Co/Silver** | 中規模 | 10 | 1 | 広範な open-vocab 評価 |
| **SA-Co/Bronze** | 既存マスクアノテーション | 9 既存データセット | 既存 | 比較用 |
| **SA-Co/Bio** | 生物学/医療系 | 既存データセット | 既存 | ドメイン特化評価 |
| **SA-Co/VEval** | 動画評価 | 3 | 1 | 動画 PCS 評価（SA-V, YT-Temporal-1B, SmartGlasses） |

### Gold で 3 アノテーター/ペアの理由

PCS は本質的に曖昧（"mouse" デバイス vs 動物等）:

- 1 アノテーターだと正解が 1 つに固定されてしまう
- 3 人で評価し、**oracle 評価**（モデル予測を全 ground truth と比較して最良スコアを取る）で公平に
- これで「複数の妥当な解釈」を許容する評価が可能に

### SA-Co Benchmark の指標

- **cgF₁** = `100 × pmF₁ × IL_MCC`（主要指標）
- **pmF₁** = positive media-phrase pair での micro F1（位置特定）
- **IL_MCC** = image-level Matthews Correlation Coefficient（認識）
- **信頼度 0.5 以上の予測のみ** で評価（キャリブレーション強制）
- 動画: **pHOTA**（positive HOTA、open-vocab tracking）

SAM 3 の SA-Co/Gold 結果：

| Model | cgF₁ | IL_MCC | pmF₁ |
|---|---|---|---|
| Human | 72.8 | 0.94 | 77.0 |
| OWLv2* | 24.6 | – | – |
| LLMDet | 6.5 | – | – |
| APE-D* | 16.4 | – | – |
| DINO-X | 21.3 | – | – |
| Gemini 2.5 | 13.0 | – | – |
| **SAM 3** | **54.1** | **0.82** | **65.9** |

**人間性能 72.8 の 74%、最強ベースライン OWLv2* の 2.2 倍**。

## SA-Co Ontology（コンセプト体系）

訓練と評価の語彙管理のため、**Wikidata ベースの 22.4M ノードの ontology** を構築：

- **17 トップレベルカテゴリ**
- **72 サブカテゴリ**
- long-tail で fine-grained なコンセプトをカバー
- AI による NP マイニングの基盤

これはコンセプトの **体系性** を保証する仕組み。任意の英語名詞句ではなく、ontology に紐付くものを優先する設計。

## 既存データセットとの比較

| データセット | 主タスク | ユニーク concept 数 | 画像数 | 動画数 | マスク数 |
|---|---|---|---|---|---|
| **SA-1B** | PVS（画像） | （ラベルなし） | 11M | – | 1.1B |
| **SA-V** | PVS（動画） | （ラベルなし） | – | 50.9K | 35.5M |
| LVIS | OV 検出/seg | 1,203 (closed) | 164K | – | 1.6M |
| Open Images | 検出/seg | 600 + 19,957 (closed) | 9M | – | 2.7M |
| ADE20K | semantic seg | 150 (closed) | 25K | – | 0.7M |
| COCO | 検出/seg | 80 (closed) | 118K | – | 0.86M |
| **SA-Co/HQ + Bench** | **PCS** | **4M (train) + 207K (bench, open)** | **5.2M + 120K** | **52.5K + 1.7K** | **52M (train)** |

→ **コンセプト数で桁違いの飛躍**。先行の最大が Open Images の ~20K に対し SA-Co は **4M（200 倍）** + ベンチマークだけで 207K。

## ライセンスと公開

- SAM 3 と同時公開予定
- ベンチマーク（SA-Co Benchmark）は **オープンソース化**（モデルと共に Apache 2.0 / CC by 4.0 想定、詳細待ち）
- 訓練データ（SA-Co/HQ）は研究目的、一部公開

## なぜ重要か

1. **Open-vocabulary の桁違いスケール**: 4M unique NP、207K benchmark concepts は分野を一段引き上げる
2. **AI verifier ベースの効率化**: Llama 3.2 で人間アノテーションを半減、scalability の証明
3. **Hard negative の標準化**: 訓練データに敵対的 negative を含める設計が普及する触媒に
4. **Ontology-driven mining**: 体系的なコンセプトカバレッジが open-vocab 評価の信頼性を上げる
5. **PCS 評価の標準**: cgF₁ + IL_MCC + pmF₁ という新指標セットが分野標準になる可能性

## 限界

1. **言語が NP に限定**: 「青いシャツの男性の右の人」のような長い指示表現はカバーしない
2. **動画は短時間（30 秒以下）**: 長尺動画理解には不十分
3. **曖昧性は緩和のみで根本解決ではない**: 多義語・主観的記述子は依然問題
4. **3 アノテーター/ペアは Gold のみ**: 他分割は 1 アノテーター → 評価のばらつきあり
5. **ライセンスや訓練データ完全公開の詳細待ち**: SAM 3 リリース時に確認必要

## 関連ページ

- [[sources/sam-3]] — SA-Co を構築した SAM 3 論文の要約
- [[entities/sam-3]] — SA-Co で訓練された / SA-Co を生成した SAM 3
- [[concepts/promptable-concept-segmentation]] — SA-Co が駆動する PCS タスク
- [[entities/sa-1b]] / [[entities/sa-v]] — SAM 1/2 の対応データセット（画像/動画）
- [[entities/wit-400m]] — CLIP の対応データセット（画像-テキスト対）
- [[entities/lvd-1689m]] — DINOv3 の対応データセット（純 SSL）
- [[overview]] — CV データセット全体俯瞰
