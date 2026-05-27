---
type: entity
entity_kind: model
aliases: [SAM 3, Segment Anything Model 3, SAM v3]
tags: [model, segmentation, concept, open-vocabulary, video, foundation-model, meta-superintelligence-labs, perception-encoder, detr]
related: [[concepts/promptable-concept-segmentation]], [[concepts/promptable-segmentation]], [[concepts/foundation-model]]
sources: [[sources/sam-3]]
updated: 2026-05-26
---

# SAM 3（Segment Anything Model 3）

## 概要

**SAM 3** = **Segment Anything Model 3** with Concepts。**Meta Superintelligence Labs** チーム（Carion ら, 2025）が発表した、画像と動画における **コンセプトベースのセグメンテーション基盤モデル**。SAM v1（[[entities/sam]]）と SAM 2（[[entities/sam-2]]）の自然な後継で、PVS タスクに加えて新タスク **PCS**（Promptable Concept Segmentation, [[concepts/promptable-concept-segmentation]]）を導入した。

- 論文: "SAM 3: Segment Anything with Concepts"
- arXiv: 2511.16719（2025 年 11 月）
- プロジェクト: 公開予定（モデル + SA-Co ベンチマーク + 推論コード）
- 詳細解説: [[sources/sam-3]] / 翻訳: [[translations/sam-3]]

> **補足: Meta Superintelligence Labs** — 従来の Meta FAIR と異なる新組織。SAM 1/2 が FAIR 発だったのに対し、SAM 3 はこの新ラボから発表された。同時期に Meta は様々な AI 研究組織を再編。

## 何が新しいか（SAM 2 → SAM 3）

| | SAM 2（2024, FAIR） | **SAM 3（2025, Superintelligence Labs）** |
|---|---|---|
| **タスク** | PVS（画像 + 動画） | **PVS + PCS（画像 + 動画）** |
| **プロンプト型** | 点、ボックス、マスク（PVS のみ） | 点、ボックス、マスク + **名詞句 (NP) + 画像 exemplar** |
| **出力** | 1 オブジェクトの masklet | 1 オブジェクト OR **コンセプトの全インスタンス**（同一性保持） |
| **画像エンコーダ** | Hiera（MAE 事前学習） | **Perception Encoder (PE)** ([[entities/perception-encoder]]) |
| **検出器** | なし | **DETR ベース** + presence head + exemplar encoder |
| **訓練データ** | SA-V + SA-1B | + **SA-Co/HQ (5.2M 画像, 4M NP, 52M マスク)** + SA-Co/SYN |
| **データ規模** | 642.6K masklet | **5.2M 画像 + 52.5K 動画 + 4M unique NP** |
| **ベンチマーク** | SA-V val/test | **SA-Co** (207K concepts、既存の 50 倍) |
| **データエンジン** | 3 phase（モデル支援） | **4 phase + AI verifiers**（Llama 3.2 ベース、スループット 2 倍） |

## アーキテクチャ

**dual encoder-decoder** 設計、検出器とトラッカーが共有 Perception Encoder backbone を持つ：

<figure>

![](../../raw/assets/sam-3/fig4-architecture.png)

<figcaption>図4（再掲）: SAM 3 アーキテクチャ。PE が画像とテキストをエンコード → fusion encoder で融合 → DETR decoder で proposal クエリ → 各マスク。Tracker は SAM 2 と同じメモリ機構。</figcaption>
</figure>

### 共有 backbone: Perception Encoder

[[entities/perception-encoder]] を採用。テキストと画像が aligned された強力な VL backbone で、detector と tracker の両方で同じ PE インスタンスを使う。

### 検出器（DETR ベース）

| サブコンポーネント | 内容 |
|---|---|
| **テキストエンコーダ** | PE のテキスト塔 |
| **画像エンコーダ** | PE の画像塔 |
| **Exemplar エンコーダ** | 画像 exemplar（ボックス + 正/負ラベル）を ROI プール + 位置/ラベル埋め込み + 小 transformer でエンコード |
| **Fusion encoder** | 画像埋め込みをプロンプトトークン（テキスト + exemplar）にクロス注意で条件付け |
| **DETR decoder** | 学習済みオブジェクトクエリが fusion 出力にクロス注意、box delta を Deformable DETR スタイルで予測 |
| **Mask head** | MaskFormer 適応 |
| **Semantic head** | 全画素にコンセプト所属の二値ラベル |
| **Box-region-positional bias** | Plain-DETR から、注意を各オブジェクトに集中させる |
| **Dual supervision** | DAC-DETR の手法 |
| **Align loss** | aligndetr 由来 |

### Presence Token（SAM 3 最大の発明）

学習されたグローバル **presence token** が認識を分離して担当：

- **proposal クエリのスコア** = $p(q_i \text{ is a match} | \text{NP が画像に存在})$
- **presence token のスコア** = $p(\text{NP が画像に存在})$
- **最終スコア** = 両者の積

これにより：
- proposal クエリは「位置特定」のみに専念（局所的判断）
- presence token は「認識」のみに専念（グローバル文脈）
- **negative 画像**ではproposal クエリに勾配を流さず、presence token のみが「ない」を学ぶ

アブレーション結果：
- presence head 追加で **cgF₁ +1.5、IL_MCC +0.05**
- hard negatives と組み合わせると IL_MCC が **0.44 → 0.68**（劇的改善）

### Tracker（SAM 2 継承）

SAM 2（[[entities/sam-2]]）と同じ memory ベース設計：
- Memory encoder + memory bank + mask decoder + prompt encoder
- 検出器と **同じ PE backbone を共有**（凍結）
- 各フレームで検出器が新オブジェクト $\mathcal{O}_t$ を見つけ、tracker が masklet $\mathcal{M}_{t-1} \to \hat{\mathcal{M}}_t$ を伝播

### Detection ↔ Tracking matching

両者の衝突を解決する 2 戦略：
1. **Masklet detection score**: 時間ウィンドウ内で masklet がどれだけ一貫して検出にマッチしたか。閾値以下で抑制
2. **周期的 detection 再プロンプト**: 高信頼度検出マスクで tracker を定期的に上書き、遮蔽や distractor 失敗を回避

## 訓練ステージ（4 段階）

1. **PE 事前学習**: Perception Encoder の VL アラインメント学習（既存）
2. **検出器事前学習**: DETR 部分を大規模で訓練
3. **検出器ファインチューニング**: SA-Co/HQ で精緻化
4. **Tracker 訓練**: backbone を凍結、SAM 2 スタイルで tracker を訓練

## モデルサイズ

論文には複数サイズの開示が少ないが、明示されているのは：

- 単一画像で **100+ オブジェクト検出を 30ms（H200 GPU）**
- 動画は同時オブジェクト数に線形依存、約 5 オブジェクトで近実時間

具体的なパラメータ数は公開待ち。

## 主要結果

### Image PCS（表 1）

| Model | LVIS cgF₁ | LVIS AP | SA-Co/Gold cgF₁ | SA-Co/Silver |
|---|---|---|---|---|
| Human | – | – | **72.8** | – |
| OWLv2* | 29.3 | 43.4 | 24.6 | 11.5 |
| LLMDet-L | 35.1 | 36.3 | 6.5 | 7.1 |
| APE-D* | – | 53.0 | 16.4 | 7.3 |
| DINO-X | – | 38.5 | 21.3 | – |
| **SAM 3** | **37.2** | **48.5** | **54.1** | **49.6** |

- LVIS マスク AP で **48.5**（DINO-X 38.5 を圧倒）
- SA-Co/Gold で **54.1**（OWLv2* 24.6 の **2.2 倍**、**人間性能の 74%**）

### Video PCS（表 4）

| Benchmark | SAM 3 cgF₁ | SAM 3 pHOTA |
|---|---|---|
| SA-V (2.0K NPs) | **30.3** | **58.0** |
| YT-Temporal-1B (1.7K NPs) | **50.8** | **69.9** |
| SmartGlasses (2.4K NPs) | **36.4** | **63.6** |

NP 数が多いほど SAM 3 の優位性が大きい（open-vocabulary 能力の証）。

### PVS（SAM 2 比較、表 5）

| Benchmark | SAM 2.1 L | **SAM 3** |
|---|---|---|
| MOSEv1 val | 77.9 | **78.4** |
| DAVIS17 val | 90.7 | **92.2** |
| LVOSv2 val | 79.6 | **88.5** |
| SA-V val | 77.9 | **83.5** |
| SA-V test | 78.4 | **84.4** |
| **MOSEv2 val** | 47.9 | **60.3** (+12.4) |

→ **PVS でも SAM 2 を凌駕**。MOSEv2 で大幅 +12.4 ポイント。

### SAM 3 Agent（複雑言語クエリ、表 7）

MLLM が SAM 3 を tool として使う pattern：

| MLLM | ReasonSeg val (gIoU) | ReasonSeg test | OmniLabel descr |
|---|---|---|---|
| – | Overall SOTA: 65.0 | 61.3 | 36.5 |
| Qwen2.5-VL 7B | 62.2 | 63.0 | 36.7 |
| Llama4 Maverick | 68.5 | 67.1 | 32.8 |
| Qwen2.5-VL 72B | 74.6 | 70.8 | 42.0 |
| **Gemini 2.5 Pro** | **77.0** | **74.0** | **45.3** |

**指示表現/推論セグメンテーションの訓練なし**で先行 SOTA を超える。

### Counting（表 3）

- CountBench MAE **0.12** / Acc 93.8%（次点 Gemini 2.5 Pro 0.24 の半分）
- マスクも返せる利点

### Few-shot Adaptation（表 2-3）

- ODinW13 10-shot AP **71.8**（gDino1.5-Pro 67.9 を超える）
- 1 exemplar PCS で T-Rex2 を COCO +18.3, LVIS +10.3, ODinW +20.5 で圧倒

## 推論コード（公開予定）

公開ライセンスは Apache 2.0 想定（SAM 1/2 と同じ）、API は SAM 2 と類似と推測されるが、PCS 用 API が追加される：

```python
# 想定 API（公開待ち）
from sam3 import Sam3Predictor

predictor = Sam3Predictor.from_pretrained("facebook/sam-3")

# PCS: 名詞句プロンプト
masks = predictor.predict_concept(image, text="yellow school bus")

# PCS: 画像 exemplar
masks = predictor.predict_concept(
    image,
    exemplars=[(box, label=1)],  # label: 1=positive, 0=negative
)

# PCS: テキスト + exemplar
masks = predictor.predict_concept(
    image,
    text="cat",
    exemplars=[(missing_cat_box, 1), (false_positive_box, 0)],
)

# PVS は SAM 2 風 API
sam2_style = predictor.predict(image, point_coords=..., point_labels=...)

# 動画 PCS
video_masklets = predictor.predict_video_concept(
    video_frames, text="all people"
)
```

## SAM 3 Agent パターン

複雑な指示表現や推論を含むクエリは MLLM 統合で扱う：

```python
# 概念モデル（実装パターン）
def sam3_agent(query: str, image, mllm):
    # 1. MLLM に NP を提案させる
    nps = mllm.propose_noun_phrases(query, image)

    # 2. 各 NP で SAM 3 にプロンプト
    candidates = [sam3.predict_concept(image, text=np) for np in nps]

    # 3. MLLM に結果を分析させ、満足なら返す、ダメなら反復
    while not mllm.is_satisfactory(candidates, query):
        nps = mllm.refine(query, candidates)
        candidates = [sam3.predict_concept(image, text=np) for np in nps]

    return mllm.select_best(candidates, query)
```

ReasonSeg / OmniLabel で SOTA を達成、これらの **指示表現データでの訓練なし**で。

## 派生・後続候補

SAM 3 が 2025/11 発表で、まだ派生は登場していないが、SAM 1/2 の歴史から予想：

| 派生種 | 期待される展開 |
|---|---|
| **MobileSAM 3 / FastSAM 3** | エッジ向け軽量版 |
| **MedSAM 3** | 医療画像/動画への適応 |
| **SAM 3 + 3D** | NeRF, Gaussian Splatting と統合 |
| **Grounded SAM 3** | より高度な grounding pipeline |
| **SAM 3 + 動画 VLM** | SAM 3 Agent の拡張、長尺動画理解 |
| **ロボット行動向け SAM 3** | PCS 出力 → 行動方策 |

## SAM 1/2/3 比較表

| | **SAM v1 (2023)** | **SAM 2 (2024)** | **SAM 3 (2025)** |
|---|---|---|---|
| 組織 | Meta FAIR | Meta FAIR | **Meta Superintelligence Labs** |
| タスク | PVS（画像） | PVS（画像+動画） | **PVS + PCS（画像+動画）** |
| プロンプト型 | 点、ボックス、マスク、テキスト(PoC) | 点、ボックス、マスク | **+ 名詞句、画像 exemplar** |
| 出力 | 1 オブジェクト | 1 オブジェクト/masklet | **コンセプトの全インスタンス** |
| 画像エンコーダ | ViT-H/16（MAE） | Hiera（MAE） | **Perception Encoder** |
| Detector | なし | なし | **DETR ベース + presence head** |
| Tracker | なし | あり（memory） | あり（SAM 2 継承） |
| 訓練データ | SA-1B（11M 画像） | + SA-V（50.9K 動画） | + **SA-Co/HQ（5.2M 画像/4M NP）** |
| ベンチマーク | 23 ゼロショット | 17 video + 37 image | **+ SA-Co（207K concepts）** |
| データエンジン phase 数 | 3 | 3 | **4** |
| AI 検証者 | なし | なし | **Llama 3.2 ベース、スループット 2 倍** |
| MLLM 統合 | なし | なし | **SAM 3 Agent パターン** |
| 速度 | 21.7 FPS（画像、ViT-H） | 130 FPS（画像、Hiera-B+） | **30ms/画像（100+ 検出、H200）** |
| ライセンス | Apache 2.0 | Apache 2.0 | Apache 2.0 想定 |
| 訓練 GPU | 256 A100 × 68h | 256 A100 × 108h | 未開示 |

## なぜ重要か

CV foundation model の **第 4 の進化**：

1. **コンセプトベース segmentation の確立**: 「インスタンス指定」から「コンセプト指定」へのパラダイムシフト
2. **テキストと画像 exemplar を統一**: 言語ベースと例示ベースの両方をサポートする初の本格モデル
3. **PVS と PCS の統一**: 単一モデルで両タスク、相補的に使える
4. **Presence head の発明**: open-vocabulary 検出での「認識 vs 位置特定」衝突を分離する一般的設計パターン
5. **AI verifier の確立**: Llama ベースの自動検証者で人間アノテーションコストを半減、合成データのスケーリング則も実証
6. **Perception Encoder の実用化**: PE backbone を foundation model で初めて本格採用、CLIP/SigLIP からの進化
7. **SAM 3 Agent パターン**: foundation model + MLLM の組み合わせで複雑タスクを解く新パターン

## 限界（§8）

1. **ドメイン外用語への弱さ**: 自動ドメイン拡張で緩和可能、ただし追加訓練必要
2. **長い指示表現/推論クエリは対応外**: SAM 3 Agent パターンで対応
3. **動詞・関係表現困難**: NP のみ、「投げている人」のような動作は捉えにくい
4. **動画オブジェクト数に線形依存**: 約 5 オブジェクトが近実時間限界
5. **本質的曖昧性**: oracle 評価で緩和するが、根本解決ではない
6. **detection/tracking 衝突の完全解決でない**: matching function で緩和

## 関連ページ

- [[sources/sam-3]] — 原典の要約
- [[translations/sam-3]] — 全文和訳（Abstract + §1-9 + Appendix A.1 部分）
- [[concepts/promptable-concept-segmentation]] — SAM 3 が定義した新タスク
- [[entities/sa-co]] — SAM 3 の訓練・評価データ
- [[entities/perception-encoder]] — SAM 3 の backbone
- [[entities/sam-2]] — 直前世代（tracker を継承）
- [[entities/sam]] — SAM v1（系列の元）
- [[entities/sa-1b]] / [[entities/sa-v]] — 先行データセット
- [[concepts/promptable-segmentation]] — PVS（SAM 3 でも継承）
- [[concepts/foundation-model]] — CV foundation model 系譜
