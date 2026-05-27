---
type: concept
aliases: [Promptable Segmentation, プロンプト可能セグメンテーション, promptable mask]
tags: [task, segmentation, foundation-model, prompt-engineering]
related: [[zero-shot-transfer]], [[foundation-model]], [[contrastive-learning]], [[promptable-concept-segmentation]]
sources: [[sources/segment-anything]], [[sources/sam-2]], [[sources/sam-3]]
updated: 2026-05-26
---

# Promptable Segmentation（プロンプト可能セグメンテーション）

## 一言で

**「任意のプロンプト（点、ボックス、粗いマスク、テキスト）が与えられたとき、妥当な（valid）セグメンテーションマスクを返すタスク」**。SAM（Segment Anything Model, [[entities/sam]] / [[sources/segment-anything]]）が定義した、セグメンテーション基盤モデル（[[concepts/foundation-model]]）のための事前学習目的にして、下流タスクへのゼロショット転移インターフェイス。

> **補足: 既存の "interactive segmentation" とどう違うか** — 古典的な対話的セグメンテーション（IFCN, RITM, FocalClick 等）は「ユーザーが点を打ち足すたびにマスクを精緻化していく」反復ループを想定し、**最終的に正解マスクへ収束する**ことを目指す。promptable segmentation は逆に、**たとえ 1 つのプロンプトしかなくても**、また **プロンプトが曖昧でも**、「少なくとも 1 つの合理的なマスク」を即座に返すことを要求する。これにより、人間ユーザーとの対話だけでなく、**他のアルゴリズム（検出器、視線追跡、テキストモデル等）からのプログラマティックなプロンプト** にも応答できるインターフェイスとなる。

## 動機

NLP の foundation model（GPT, BERT, PaLM 等）は「**次トークン予測 → プロンプティングで多様な下流タスク**」というパターンで成功した。CV では CLIP（[[entities/clip]]）が画像分類でこれを実現したが、**セグメンテーションのような密予測タスク** に同じ枠組みを持ち込むには、新しい「タスク定義」が必要だった。

SAM 論文（§2）の問い：**「ゼロショット汎化を可能にする pre-training タスクは何か？」**

著者の解答が promptable segmentation。これが満たすべき条件は：

1. **自然な事前学習アルゴリズムをもたらす**（プロンプトを訓練中にシミュレートして損失計算できる）
2. **広範な下流タスクへのプロンプトエンジニアリング転移を可能にする**
3. **曖昧性を許容する**（NLP の LM が曖昧な質問にも整合した応答を返すように）

## 定義の核心：「妥当な（valid）マスク」

最大の設計判断は、**「プロンプトが曖昧でも、複数の妥当オブジェクトのうち少なくとも 1 つに対する合理的マスクを返せばよい」** という緩い妥当性条件。

例: シャツ上の 1 点 → 「シャツ」「人物」「シャツの一部（ロゴなど）」のいずれかを返せば妥当。

これを実装するため SAM は **3 つのマスクを同時予測**（whole / part / subpart の入れ子）し、訓練時は最小損失のマスクからのみ逆伝播する（multi-output learning）。

## プロンプトの種類

| 種類 | 例 | エンコーディング |
|---|---|---|
| **疎: 点** | 前景/背景クリック | 位置エンコーディング + 学習埋め込み |
| **疎: ボックス** | バウンディングボックス | 左上+右下隅の位置エンコーディング + 学習埋め込み |
| **疎: テキスト** | "a wheel", "cat" | CLIP テキストエンコーダ（[[entities/clip]]） |
| **密: マスク** | 粗いマスク（前回の予測等） | 畳み込みで埋め込み → 画像埋め込みに加算 |

これらは合成できる。例: ボックス + 修正点で対話的精緻化。

## なぜプロンプト設計が中心になったか

CV の歴史的なセグメンテーションは **タスクごとに固定**：

- **semantic seg**: 各画素にクラスラベル
- **instance seg**: 個別オブジェクトごとにマスク
- **panoptic seg**: 上記の統合
- **interactive seg**: ユーザー入力で精緻化
- **edge detection**: 境界画素のみ
- **object proposal**: マスクのランク付き候補

SAM はこれらを **プロンプトを変えるだけ** で実装する：

| 下流タスク | プロンプト設計 |
|---|---|
| **対話的セグメンテーション** | ユーザーが点を順次追加 |
| **インスタンスセグメンテーション** | 検出器ボックスを入力 |
| **オブジェクト提案** | 64×64 グリッド点 → NMS |
| **エッジ検出** | 16×16 グリッド点 → Sobel フィルタ → エッジ NMS |
| **テキスト → マスク** | CLIP テキスト埋め込み（訓練時は画像埋め込みを代用） |
| **自動全マスク生成（SA-1B）** | 32×32 グリッド点 → 信頼度+安定性フィルタ |

> **補足: prompt engineering（[[concepts/zero-shot-transfer]]）の CV 版** — CLIP がテキスト分類で "a photo of a {class}" のようなプロンプトを使うのと並列の概念。SAM のプロンプトは画像空間で行われる点が違うが、**「タスクをプロンプトに翻訳することで単一モデルを多用途化する」** という発想は完全に一致する。

## 学習アルゴリズム（事前学習）

SAM は対話的セグメンテーションをシミュレートして訓練：

1. マスクごとに、最初のプロンプトとして前景点か正解ボックスを等確率で選ぶ
2. 予測を生成
3. 予測と正解のエラー領域から次の点をサンプル（偽陽性なら背景点、偽陰性なら前景点）
4. **11 イテレーション**繰り返す（1 + 8 反復点 + 2 マスクのみ）
5. **マスクあたり 11 ラウンド**を 90k イテレーション × バッチ 256 で訓練

損失は **focal loss + dice loss を 20:1 で結合**（DETR 系の標準）+ IoU 予測ヘッドへの MSE。

## 限界（SAM 論文 §8 + 派生研究）

1. **セマンティック / パノプティックセグメンテーションのプロンプト設計が不明** — SAM 自身も §8 で「これらをどうプロンプトで表現するかは未解決」と認めている
2. **modal vs. amodal の選択不可** — データセット規約に応じた選択は zero-shot では学習できない
3. **微細構造の見逃し** — FocalClick のように「ズームイン」する手法には敵わない
4. **高 IoU 領域では専用対話的手法に劣る** — 多数点プロンプトでは SimpleClick 等が勝る

## PVS（動画への拡張）

SAM 2（[[sources/sam-2]] / [[entities/sam-2]]）が定義した **Promptable Visual Segmentation** タスクは、promptable segmentation の自然な動画拡張：

- **入力**: 動画の任意フレームでの点・ボックス・マスクプロンプト
- **出力**: 動画全体での **masklet**（時空間マスク、各フレームの 2D マスクの系列）
- **対話的修正**: 別フレームで追加プロンプト → masklet を反復洗練
- **画像 = 1 フレーム動画**: 完全に統一されたインターフェイス

PVS は半教師あり VOS（第 1 フレームのマスクのみ）と対話的 VOS（複数フレームのスクリブル）を特殊ケースとして包含する。

> **補足: なぜ PVS は独立 concept にしなかったか** — SAM 2 自身が「画像 = 1 フレーム動画」と統一しており、PVS は promptable segmentation の自然な拡張に過ぎない。タスクの本質的差分は「時間メモリの有無」のみで、これは [[entities/sam-2]] のアーキテクチャ詳細に属する。**concept レベルでは promptable segmentation 1 つで十分**という設計判断。

## PCS（コンセプトプロンプト軸への拡張）

SAM 3（[[sources/sam-3]] / [[entities/sam-3]]）が 2025 年に導入した **PCS（Promptable Concept Segmentation, [[concepts/promptable-concept-segmentation]]）** は、promptable segmentation の **第 2 の直交軸**：

| | **PVS**（SAM 1/2/3） | **PCS**（SAM 3 新規） |
|---|---|---|
| **プロンプト型** | 点、ボックス、マスク（インスタンス指定） | 名詞句、画像 exemplar（コンセプト指定） |
| **出力** | 1 オブジェクト/masklet | コンセプトに合う **全インスタンス** |

PVS と PCS は **互補的**：
- PCS で「画像中のすべての犬」を取得
- PVS で個別の犬の境界を点プロンプトで洗練

SAM 3 は両方を統一モデルでサポートし、ユーザーは状況に応じてプロンプト型を切り替えられる。

> **補足: PVS の延長線か新タスクか** — PCS は概念レベルで PVS と根本的に異なる（入力型 + 出力の数）。SAM 3 論文も「PVS は単一インスタンス、PCS はコンセプトの全インスタンス」と明確に区別している。このため [[concepts/promptable-concept-segmentation]] は **独立 concept** に昇格させた（PVS が SAM 2 で「画像 = 1 フレーム動画」として PVS の自然な拡張だったのとは対照的）。

## 後続研究

| 系統 | 例 | 拡張点 |
|---|---|---|
| **動画拡張** | **SAM 2**（Meta, 2024, [[entities/sam-2]] / [[sources/sam-2]]） | **PVS（Promptable Visual Segmentation）** タスクとして動画に一般化。streaming memory（空間 + プロンプト + object pointer）+ Hiera エンコーダ。画像は 1 フレーム動画として統一処理 → SAM v1 比 6× 高速 + 高精度 |
| **テキスト強化** | Grounded-SAM, Grounded-DINO + SAM | テキスト → ボックス → SAM のパイプライン |
| **高速化** | MobileSAM, FastSAM, EfficientSAM | 画像エンコーダの軽量化（蒸留・ConvNet 代替） |
| **品質改善** | HQ-SAM | 高品質マスク用の追加トークン |
| **ドメイン適応** | MedSAM, SatSAM | 医療画像、衛星画像へのファインチューン |
| **3D 拡張** | SAM 3D, SegmentAnything3D | 点群やメッシュへのプロンプト可能化 |

## SAM の前にもプロンプト可能セグメンテーションはあったか？

部分的にはあった：

- **interactive segmentation** 全般（Snake (1988), GrabCut (2004), IFCN (2016) など）は「ユーザー入力 → マスク」という意味でプロンプト可能だった
- ただし **「曖昧プロンプトに対する複数候補出力」「画像エンコーダの一度きり計算による償却的実時間性」「自由形式テキスト/ボックスの統一インターフェイス」「単一モデルでの多タスク化」** は SAM 固有
- 重要なのは「**プロンプト可能タスクを foundation model の pre-training objective にした**」点

## 関連ページ

- [[sources/segment-anything]] — SAM v1 原典の要約ページ
- [[sources/sam-2]] — SAM 2 原典の要約（PVS = 動画版）
- [[sources/sam-3]] — SAM 3 原典の要約（PCS の導入元）
- [[concepts/promptable-concept-segmentation]] — PCS タスク（PVS と互補的な軸）
- [[entities/sam]] / [[entities/sam-2]] / [[entities/sam-3]] — SAM 系列モデルのスペック
- [[entities/sa-1b]] / [[entities/sa-v]] / [[entities/sa-co]] — promptable segmentation を駆動するデータセット（画像/動画/コンセプト）
- [[entities/hiera]] — SAM 2 の画像エンコーダ
- [[entities/perception-encoder]] — SAM 3 の画像エンコーダ
- [[concepts/zero-shot-transfer]] — プロンプトエンジニアリングを介したゼロショット転移
- [[concepts/foundation-model]] — CV foundation model 全体像
- [[entities/clip]] — SAM v1 のテキストプロンプトで使う CLIP テキストエンコーダ
- [[entities/mae]] — SAM v1 と SAM 2 の画像エンコーダ初期化に使う MAE
