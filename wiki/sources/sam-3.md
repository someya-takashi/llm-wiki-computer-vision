---
type: source
source_path: raw/papers/SAM 3_ Segment Anything with Concepts.md
source_kind: paper
title: "SAM 3: Segment Anything with Concepts"
authors: [Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, Jie Lei, Tengyu Ma, Baishan Guo, Arpit Kalla, Markus Marks, Joseph Greer, Meng Wang, Peize Sun, Roman Rädle, Triantafyllos Afouras, Effrosyni Mavroudi, Katherine Xu, Tsung-Han Wu, Yu Zhou, Liliane Momeni, Rishi Hazra, Shuangrui Ding, Sagar Vaze, Francois Porcher, Feng Li, Siyuan Li, Aishwarya Kamath, Ho Kei Cheng, Piotr Dollár, Nikhila Ravi, Kate Saenko, Pengchuan Zhang, Christoph Feichtenhofer]
year: 2025
venue: arXiv 2511.16719
ingested: 2026-05-26
tags: [sam-3, segmentation, concept, open-vocabulary, foundation-model, meta-superintelligence-labs, detr, perception-encoder]
translation: [[translations/sam-3]]
---

# SAM 3: Segment Anything with Concepts

> 原典: [[translations/sam-3]] ・ `raw/papers/SAM 3_ Segment Anything with Concepts.md`
> 著者: Carion, Gustafson, Hu, Debnath, Hu, Suris, Ryali, Alwala, Khedr, Huang, Lei, Ma, Guo, Kalla, Marks, Greer, Wang, Sun, Rädle, Afouras, Mavroudi, Xu, Wu, Zhou, Momeni, Hazra, Ding, Vaze, Porcher, Li, Li, Kamath, Cheng, Dollár, Ravi, Saenko, Zhang, Feichtenhofer（**Meta Superintelligence Labs**）
> 発表: arXiv:2511.16719（2025 年 11 月）

> **注記**: 原典ファイル（raw/papers/SAM 3_ Segment Anything with Concepts.md）には Appendix の実体が含まれないため、本要約は **Abstract + §1-9 + Appendix A.1 冒頭（Presence Token 部分）** に基づく。Appendix の残り部分（A.2 以降、B-H）には言及できない。

## 一言まとめ

**SAM の第 3 世代**。SAM 2（[[sources/sam-2]]）が masklet 単位での「動画 PVS」を解いたのに対し、SAM 3 は **「短い名詞句や画像 exemplar を与えると、コンセプトに合うすべてのインスタンスを検出・セグメント・追跡する」** という新タスク **PCS（Promptable Concept Segmentation）** を導入。**DETR ベース検出器 + SAM 2 風 tracker** が共有 **Perception Encoder backbone** を持つ統一アーキテクチャ、認識と位置特定を分離する **presence head** 設計、AI 検証者を活用するデータエンジンで **4M unique NP + 52M マスク** の SA-Co/HQ と **207K コンセプト**（既存の 50 倍）の SA-Co ベンチマークを構築。**LVIS ゼロショットマスク AP 48.8（先行 38.5 を圧倒）**、SA-Co で先行ベースライン比 2 倍以上。

## 背景と問題意識

### SAM v1/v2 の根本的限界：「インスタンス指定」しかできない

[[entities/sam]] と [[entities/sam-2]] はいずれも **「1 プロンプト → 1 オブジェクト」** の枠組み（PVS, [[concepts/promptable-segmentation]]）。点・ボックス・マスクで「この 1 つの物体」を指せるだけで、**「画像/動画中のすべての犬」「すべての黄色いスクールバス」** を一度に取り出すには別のモデル（OWLv2, GroundingDINO, GLEE 等）が必要だった。

CV 応用としてしばしば必要なのは：
- データアノテーション: 「全インスタンスのマスクが欲しい」
- ロボティクス: 「シーン中のすべての赤いボウル」
- 動画編集: 「動画を通してすべての車を追跡」
- AR/VR: 「視野中のすべてのテキスト」

**「コンセプトを指定して全インスタンスを得る」** という基本タスクが SAM 系で未対応だった。SAM 1 §7.5 にテキスト → マスクの proof-of-concept があったが、SAM 2 では削除されていた。SAM 3 はこのギャップを正面から埋める。

### 既存 open-vocabulary 検出器（OWLv2, gDino 等）の弱点

CLIP テキストエンコーダ + 検出ヘッドというパターン（OWLv2, GroundingDINO, LLMDet）は存在したが：

1. **キャリブレーションが悪い**: ベースラインが SA-Co で cgF₁ 一桁台しか取れない
2. **対話的洗練がない**: 結果が悪くてもプロンプト追加で改善できない
3. **動画対応なし**: 追跡能力なし
4. **マスク品質が低い**: ボックス検出器 + SAM のような pipeline で繋ぐ必要があった

SAM 3 はこれらを単一モデルで一気に解く。

## 提案手法

### 1. PCS（Promptable Concept Segmentation）タスク

**入力**:
- **短い名詞句（NP）**: "yellow school bus", "striped cat" など。修飾語 + 名詞の単純構造に制限
- **画像 exemplar**: バウンディングボックス + 正/負ラベル
- **両方の組み合わせ**

**出力**:
- マッチするすべての **インスタンス** のマスク + 一意の ID
- 動画では masklet（時空間マスク）として全フレーム追跡

**対話性**: 結果を見て exemplar や NP を追加・更新して洗練可能。視覚プロンプト（点・ボックス）で個別 masklet を SAM 2 スタイルで洗練もできる。

> **補足: なぜ「単純名詞句」に限定するか** — 「青いシャツを着た男性の右側にいる帽子の人」のような長い指示表現や、「最も大きい物体」のような推論を必要とする query は SAM 3 のスコープ外。これらは「**SAM 3 Agent**」（§6, MLLM で SAM 3 をツールとして使う）で対応する設計判断。**コンセプト認識と推論を分離** して、原子的なコンセプトの認識をシンプルに保つ。

PCS は [[concepts/promptable-segmentation]] と直交する第 2 のプロンプト軸：

| | プロンプト | 出力 |
|---|---|---|
| **PVS**（SAM 1/2） | 点・ボックス・マスク（1 インスタンス指定） | 1 オブジェクトのマスク |
| **PCS**（SAM 3 新規） | 名詞句・画像 exemplar（コンセプト指定） | コンセプトに合う全インスタンスのマスク |

SAM 3 は **両方を統一モデルで** サポートし、互いに補完的に使える。

### 2. アーキテクチャ：dual encoder-decoder（detector + tracker）

<figure>

![](../../raw/assets/sam-3/fig4-architecture.png)

<figcaption>図4（再掲）: SAM 3 アーキテクチャ概要。Perception Encoder backbone を検出器とトラッカーで共有。検出器は DETR 系で、テキスト + 画像 exemplar をプロンプトトークンとして fusion encoder で融合する。tracker は SAM 2 のメモリベース設計を継承。</figcaption>
</figure>

| コンポーネント | 構成 | 役割 |
|---|---|---|
| **共有 backbone** | **Perception Encoder (PE)**（[[entities/perception-encoder]]） | テキスト ↔ 画像が aligned された強力な VL backbone |
| **検出器** | DETR ベース（M)DETR/Deformable DETR/Plain DETR の系譜）、fusion encoder で画像 ↔ プロンプトトークンを融合 → DETR decoder + マスクヘッド | コンセプトプロンプトから画像全フレームの instance マスクを生成 |
| **Presence head**（新規） | グローバル presence token、p(NP が入力に存在) のみを予測 | **認識と位置特定を分離**（最重要設計） |
| **Exemplar encoder** | ROI プール + 位置/ラベル埋め込み + 小 transformer | 画像 exemplar をプロンプトトークンに変換 |
| **Tracker** | SAM 2 と同じ（memory encoder + memory bank + mask decoder） | masklet を伝播、検出器とマッチして更新 |
| **Matching function** | IoU ベース + masklet detection score + 周期的 detection 再プロンプト | 検出器（identity 不可知）とトラッカー（identity 保持）の衝突を解決 |

### 3. Presence head の核心（SAM 3 の最大の発明）

各 proposal クエリ $q_i$ に「**何を**（recognition）」と「**どこに**（localization）」の両方を背負わせると失敗する：

- **認識** = 画像全体の文脈が必要（"cat" がこの画像にあるか）
- **位置特定** = 局所的判断（このボックスは cat 形状か）

この衝突を解消するため、グローバル **presence token** を 1 つ導入：

- presence token のスコア $p(\text{NP is present})$ = グローバル認識のみを担当
- 各 proposal クエリのスコア $p(q_i \text{ is a match} | \text{NP present})$ = 局所位置特定のみ
- **最終スコア = presence × proposal score**

これは **「open-vocabulary 検出での recognition vs localization 衝突」を根本から分離する設計**。アブレーション（表 8a）で **cgF₁ +1.5、IL_MCC +0.05** の改善。Hard negative 訓練と組み合わせると IL_MCC が 0.44 → 0.68 という劇的改善。

> **補足: 既存 DETR との違い** — 通常の DETR では「概念が画像に存在しない」場合、すべての proposal クエリに negative ラベルを与えて教師あり学習する（全クエリが「これは違う」を学ぶ）。SAM 3 では概念が画像レベルで negative なら proposal クエリには勾配を流さず、presence token のみが「ない」を学ぶ。「位置特定は条件付き確率に絞り、認識はグローバルに専門化する」という分業の徹底。

### 4. データエンジン（4 phase, AI verifiers が鍵）

SAM 1/2 の「モデル支援アノテーション」をさらに進化：

| Phase | 内容 | 主要新規 | 収集 |
|---|---|---|---|
| **1: Human Verification** | SAM 2 + 検出器でマスク提案 → 人間が MV (Mask) + EV (Exhaustivity) 検証 | ベースライン | 4.3M image-NP |
| **2: Human + AI Verification** | **Llama 3.2 を MV/EV にファインチューン → AI 検証者化**。NP 生成も Llama で hard negative 含む | **AI verifier**（人間に近い精度、スループット 2 倍） | +122M image-NP |
| **3: Scaling + Domain Expansion** | 22.4M ノードの SA-Co ontology（Wikidata ベース）から long-tail NP マイニング、15 ドメインへ拡張 | **ontology-driven NP mining** | +19.5M image-NP |
| **4: Video Annotation** | 成熟した image SAM 3 を動画に拡張、シーン/動きフィルタ、混雑/失敗シーン優先 | **動画への拡張** | 52.5K 動画 / 467K masklet |

### 5. SA-Co データセット

**訓練**:
- **SA-Co/HQ**: 5.2M 画像 + 4M unique NP（**最大の高品質 open-vocab セグメンテーションデータ**）
- **SA-Co/SYN**: 38M 句 + 1.4B マスク（合成、人間なし）
- **SA-Co/EXT**: 15 外部データセットに hard negative を強化
- **SA-Co/VIDEO**: 52.5K 動画 + 24.8K unique NP + 134K video-NP（平均 84.1 フレーム / 6 fps）

**ベンチマーク**:
- **207K unique concepts** in 120K 画像 + 1.7K 動画
- 既存ベンチマークの **50 倍以上のコンセプト**
- 4 分割: **Gold**（7 ドメイン、3 アノテーター/ペア）、**Silver**（10 ドメイン、1 アノテーター）、**Bronze + Bio**（9 既存データセット）、**VEval**（動画）

詳細は [[entities/sa-co]] 参照。

### 6. 評価メトリクスの工夫

PCS は「インスタンス検出 + 認識」の合成タスクなので、**キャリブレーション** が重要：

- **AP のような指標は信頼度閾値に不可知** → 実用で「閾値どうする？」問題が残る
- → **信頼度 0.5 を超える予測のみ** で評価する設計
- **cgF₁**（classification-gated F1）= `100 × pmF₁ × IL_MCC`
  - **pmF₁** = positive media-phrase pair（ground-truth ありペア）での micro F1（位置特定品質）
  - **IL_MCC** = image-level Matthews Correlation Coefficient（"NP がこの画像にあるか" の二値分類精度、$[-1, 1]$）

これにより「マスク品質も認識精度もキャリブレーションも全部測る」単一指標が成立。

## 実験結果と知見

### Image PCS with Text（表 1）

| Model | LVIS cgF₁ | LVIS AP | SA-Co/Gold cgF₁ |
|---|---|---|---|
| Human | – | – | 72.8 |
| OWLv2 | 20.1 | – | 17.3 |
| OWLv2* | 29.3 | 43.4 | 24.6 |
| LLMDet-L | 35.1 | 36.3 | 6.5 |
| APE-D* | – | 53.0 | 16.4 |
| DINO-X | – | 38.5 | 21.3 |
| Gemini 2.5 | 13.4 | – | 13.0 |
| **SAM 3** | **37.2** | **48.5** | **54.1** |

- **LVIS マスク AP で 48.5**（DINO-X 38.5 を圧倒）
- **SA-Co/Gold で 54.1**（OWLv2* 24.6 の **2.2 倍**）、**人間性能 72.8 の 74%**
- Open-vocabulary semantic segmentation（ADE-847 / PascalConcept-59 / Cityscapes）でも APE を上回る

### Video PCS（表 4）

| Benchmark (NP 数) | Baseline 最良 | SAM 3 | Human |
|---|---|---|---|
| SA-V (2.0K NPs) | SAM 3 Det+T-by-D 25.7 | **30.3** | 53.1 |
| YT-Temporal-1B (1.7K NPs) | 47.6 | **50.8** | 71.2 |
| SmartGlasses (2.4K NPs) | 29.7 | **36.4** | 58.5 |
| BURST (482 NPs) | – | **44.5 HOTA** | – |

NP 数が多いほど SAM 3 の優位性が大きい（open-vocabulary 能力の証）。

### PVS（SAM 2 比較、表 5）

SAM 3 は PVS でも SAM 2 を凌駕：

| Benchmark | SAM 2.1 L | **SAM 3** |
|---|---|---|
| MOSEv1 val | 77.9 | **78.4** |
| DAVIS17 val | 90.7 | **92.2** |
| LVOSv2 val | 79.6 | **88.5** |
| SA-V val | 77.9 | **83.5** |
| SA-V test | 78.4 | **84.4** |
| **MOSEv2 val** | 47.9 | **60.3**（+12.4） |

特に **MOSEv2 で +12.4 ポイント** — 困難な VOS シナリオで大幅改善。

### SAM 3 Agent（複雑言語クエリ、表 7）

MLLM が SAM 3 を tool として使う：MLLM が NP を提案 → SAM 3 がマスクを返す → MLLM が分析 → 反復。

- **ReasonSeg val 77.0**（Gemini 2.5 Pro 使用）、先行 SOTA 65.0 を超える
- **OmniLabel 45.3**、先行 36.5 を超える
- **指示表現セグメンテーション/推論セグメンテーションデータでの訓練なしで** 達成

→ SAM 3 は「コンセプト認識 + 推論」を分離する MLLM 統合パターンの基盤になる。

### Counting（表 3）

- CountBench MAE **0.12**（次点 Gemini 2.5 Pro 0.24 の半分）+ 93.8% Acc
- MLLM のように数を答えるだけでなく **マスクも返せる**

### スケーリング則 & ドメイン適応（図 9）

- データ量（SYN, HQ 両方）でべき乗則的にスケール
- **新ドメイン**（例: Food&drink）に対し、**SYN（人間 0）が HQ（人間 + AI）に追いつく**

→ 「**AI verifier + 合成データ** で新ドメイン展開が人間不要でスケールする」が実証された。これは foundation model 時代の **scalable domain expansion** の鍵。

### 推論速度

- **H200 GPU で 100+ オブジェクト検出/画像を 30ms**
- 動画は同時オブジェクト数に依存、約 5 オブジェクトで近実時間

## なぜ重要か（CV 史上の位置づけ）

### SAM 系列の進化

| | SAM v1 (2023) | SAM 2 (2024) | **SAM 3 (2025)** |
|---|---|---|---|
| タスク | PVS（画像） | PVS（画像+動画） | **PVS + PCS（画像+動画）** |
| プロンプト | 点/ボックス/マスク（テキストは PoC） | 点/ボックス/マスク | **+ 名詞句 + 画像 exemplar** |
| 出力 | 1 オブジェクト | 1 オブジェクト/masklet | **コンセプトの全インスタンス** |
| 画像エンコーダ | ViT-H/16（MAE 事前学習） | Hiera（MAE 事前学習） | **Perception Encoder** |
| データ | SA-1B（11M 画像/1.1B マスク） | + SA-V（50.9K 動画/642.6K masklet） | + **SA-Co/HQ（5.2M 画像/52M マスク/4M unique NP）** |

### CV foundation model の 3 系統 + 新軸

これまで CV foundation model は 3 系統:
- **WSL**: CLIP / SigLIP / Perception Encoder
- **SSL**: DINOv2/v3, MAE, iBOT
- **Promptable Visual**: SAM, SAM 2

SAM 3 はこれに **第 4 の軸**: **Promptable Concept**（コンセプト指定で全インスタンス取得）を加える。同時に **WSL 系（PE）と Promptable 系（SAM 2 tracker）の融合** でもある。

### 産業応用への波及

- **データアノテーション**: 「全インスタンスにマスク」が NP 1 つで取れる → アノテーション工数の革命
- **動画 VLM の vision tower**: SAM 3 Agent パターンが MLLM の標準ツールになる可能性
- **ロボティクス**: 「シーン中のすべての X」が単一モデルで取れる
- **モデルの学習データ**: AI verifier + 合成データの組み合わせは他 CV タスクに展開可能

## 限界（§8 + 関連研究）

1. **ドメイン外用語への汎化が弱い**: 著者自身が §8 で明言。Auto domain expansion で緩和可能だが追加訓練が必要
2. **長い指示表現や推論を必要とするクエリは対応外**: 「青いシャツの男性の右の人」のような複合表現は SAM 3 Agent（MLLM 統合）が必要
3. **動画推論はオブジェクト数に線形依存**: 約 5 オブジェクトで近実時間が限界
4. **テキストプロンプトが NP に限定**: 動詞や副詞の関係を表現できない
5. **Ambiguity の根本解決はできない**: 「mouse がデバイスか動物か」は 3 アノテーター + oracle 評価で緩和するが、根本的に曖昧
6. **detection/tracking 衝突の完全解決ではない**: matching function + masklet detection score で緩和するが、混雑シーンで依然失敗

## 用語と略称

- **SAM 3** = Segment Anything Model 3（[[entities/sam-3]]）
- **PCS** = Promptable Concept Segmentation（[[concepts/promptable-concept-segmentation]]、新規 concept）
- **SA-Co** = Segment Anything with Concepts dataset/benchmark（[[entities/sa-co]]）
- **NP** = Noun Phrase（短い名詞句、修飾子 + 名詞）
- **hard negative** = 敵対的に困難な negative 句（モデルが間違えやすい）
- **exemplar** = 画像 exemplar（ボックス + 正/負ラベル）
- **PE** = Perception Encoder（[[entities/perception-encoder]]、SAM 3 の backbone）
- **DETR** = DEtection TRansformer（オブジェクト検出 transformer の系譜）
- **(M)DETR** = Modulated DETR（テキスト条件付き DETR）
- **Plain-DETR** = Box-region-positional bias を持つ DETR variant
- **MaskFormer** = マスクヘッドの源流
- **presence token / presence head** = 認識専用のグローバルトークン（SAM 3 新規）
- **MV** = Mask Verification（マスク品質検証）
- **EV** = Exhaustivity Verification（網羅性検証）
- **AI verifier** = ファインチューンされた Llama 3.2 で MV/EV を実行
- **cgF₁** = classification-gated F1（SAM 3 主要メトリクス）
- **pmF₁** = positive media-phrase F1（位置特定品質）
- **IL_MCC** = Image-Level Matthews Correlation Coefficient（認識品質）
- **pHOTA** = positive HOTA（動画 PCS の主要メトリクス）
- **HOTA** = Higher Order Tracking Accuracy
- **masklet** = 動画全体の時空間マスク
- **T-by-D** = Tracking-by-Detection
- **SA-Co/HQ/SYN/EXT/Gold/Silver/Bronze/Bio/VEval/VIDEO** = SA-Co の各分割
- **SA-Co ontology** = Wikidata ベースの 22.4M ノードのコンセプト体系
- **ROI pool** = Region of Interest Pooling
- **Llama 3.2** = Meta の LLM（AI verifier のベース）

## 関連ページ

- [[translations/sam-3]] — Abstract + §1-9 + Appendix A.1 の和訳（Appendix 残部は原典に含まれず）
- [[entities/sam-3]] — SAM 3 モデルファミリーのスペック
- [[entities/sa-co]] — SA-Co データセット + benchmark の詳細
- [[concepts/promptable-concept-segmentation]] — PCS タスクパラダイム（新規）
- [[sources/segment-anything]] — SAM v1 の原典
- [[sources/sam-2]] — SAM 2 の原典（tracker と PVS の源）
- [[entities/sam]] / [[entities/sam-2]] — 前世代モデル
- [[concepts/promptable-segmentation]] — PVS（SAM 3 でも継承）
- [[entities/perception-encoder]] — SAM 3 の backbone
- [[entities/sa-1b]] / [[entities/sa-v]] — 先行データセット
- [[concepts/foundation-model]] — CV foundation model 系譜
- [[overview]] — CV 全体俯瞰における SAM 3 の位置づけ
