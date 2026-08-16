---
type: concept
aliases: [JEPA, Joint-Embedding Predictive Architecture, I-JEPA, V-JEPA]
tags: [paradigm, pretraining, ssl, world-model, lecun]
related: ["[[self-supervised-learning]]", "[[masked-image-modeling]]", "[[denoising-autoencoder]]", "[[vision-transformer]]", "[[diffusion-model]]"]
sources: ["[[sources/i-jepa]]"]
updated: 2026-06-17
---

# Joint-Embedding Predictive Architecture（JEPA, 結合埋め込み予測アーキテクチャ）

## 一言で

**「入力の一部を見て、欠けている部分を *画素ではなく抽象表現空間で* 予測する」自己教師あり学習アーキテクチャ**。Yann LeCun が「自律的機械知能への道（A Path Towards Autonomous Machine Intelligence, 2022）」で提唱した **世界モデル（world model）** 構想の中核をなす設計思想で、画像版が **I-JEPA**（[[sources/i-jepa]]）、動画版が V-JEPA / V-JEPA 2。生成モデル（画素再構成）でも対比学習（不変性）でもない、SSL の「第 3 の道」として位置づけられる。

> **補足: 世界モデル（world model）とは** — エージェントが「いま観測している状況の続きや欠落部分がどうなるか」を内部で予測するモデル。LeCun は、知能の本質は「すべてのピクセルを正確に予測する」ことではなく「**予測に値する抽象レベルで予測する**」ことだと主張し、その実装枠組みとして JEPA を提案した。

## なぜ生まれたか：生成モデルへの不満

LeCun の問題意識は「**画素を正確に予測させると、モデルは本質的に予測不可能な低レベル詳細（葉の 1 枚 1 枚、ノイズ、背景）に容量を浪費する**」というもの。

- **生成型（MAE / BEiT など、[[masked-image-modeling]]）**: マスクした領域の画素・トークンを再構成。予測不能な詳細まで当てさせられるため、表現の意味レベルが下がりやすい。
- **対比型（SimCLR / DINO など、[[self-supervised-learning]]）**: 手作業データ拡張で「不変であるべきビュー」を人間が定義。意味的だがバイアスが強く、拡張設計に依存し、他モダリティに一般化しにくい。
- **JEPA の答え**: 予測を **学習されたターゲットエンコーダの出力（抽象表現）** に対して行う。ターゲットエンコーダが「予測しても無意味な詳細」をあらかじめ捨てるので、予測器は **意味のある構造だけ** を学べばよい。手作業データ拡張も不要。

## 3 アーキテクチャの対比（EBM の枠組み）

I-JEPA 論文（[[sources/i-jepa]]）は Energy-Based Model（EBM）で 3 系統を統一的に並べる。

| | Joint-Embedding（JEA） | Generative | **JEPA** |
|---|---|---|---|
| 代表 | SimCLR, DINO, iBOT | MAE, BEiT | I-JEPA, V-JEPA |
| 何を比較するか | 2 ビューの埋め込み | 再構成画素 vs 原画素 | 予測表現 vs ターゲット表現 |
| 損失空間 | 埋め込み空間 | **入力（画素）空間** | **埋め込み空間** |
| データ拡張 | 必須（crop, color 等） | ほぼ不要 | 不要（マスクのみ） |
| 崩壊回避 | 対比/クラスタ/非対称 | 不要（再構成ターゲット） | 非対称（EMA ターゲット） |
| 条件付け変数 z | なし | マスク・位置トークン | マスク・位置トークン |

- **JEA と JEPA の違い**: JEA は「x と y を別々にエンコードして似せる」だけ（z なし、予測器なし）。JEPA は「x から **z に条件付けて** y の表現を **予測** する」（予測器あり）。
- **Generative と JEPA の違い**: 構造はそっくり（x-エンコーダ → 予測器/デコーダ → ターゲット）だが、**損失を画素で取るか表現で取るか** が決定的。I-JEPA はこれを画素にすると ImageNet-1% が 66.9 → 40.7 に激減することを実証した（[[sources/i-jepa]] 表7）。

## 崩壊（collapse）をどう防ぐか

JEPA は JEA 同様「すべての入力に同じ表現を返す」自明解（崩壊）に落ちうる。I-JEPA の対策は **x-エンコーダ（コンテキスト）と y-エンコーダ（ターゲット）の非対称性**：

- ターゲットエンコーダの重みを、コンテキストエンコーダの **EMA（指数移動平均）** で更新する。
- これは [[entities/byol]] / [[entities/dino]] の momentum encoder と同じ仕掛けで、ターゲットを滑らかに安定化し、勾配が両枝に等しく流れることによる崩壊を防ぐ。

## JEPA 系の系譜

- **I-JEPA**（Assran et al., CVPR 2023, [[sources/i-jepa]] / [[entities/i-jepa]]）: 画像 + ViT + multi-block マスキング。JEPA の最初の本格実装。
- **V-JEPA**（Bardes et al., 2024）: 動画へ拡張。時間方向のマスク予測。LeCun 構想の本命である「時間的予測」に踏み込む。（本 wiki 未取り込み）
- **V-JEPA 2**（2025）: 大規模動画 + ロボット制御への接続。物理世界の予測モデルへ。（本 wiki 未取り込み）

> **位置づけ**: 画像 SSL の主流（DINOv2/DINOv3 のハイブリッド路線、[[concepts/self-supervised-learning]]）が「最強の凍結特徴量」を競うのに対し、JEPA 系は「**世界モデル / 効率 / マルチモーダル一般化**」を狙う別系統。I-JEPA の凍結特徴量は DINOv2 ほど強くないが、低レベル・密予測タスクの汎用性と計算効率で独自の価値を主張する。

## 関連ページ

- [[sources/i-jepa]] / [[entities/i-jepa]] — JEPA の画像実装、本概念の主要文献
- [[concepts/self-supervised-learning]] — SSL 全体の中での JEPA の位置（対比型・MIM と並ぶ第 3 系統）
- [[concepts/masked-image-modeling]] — 「画素を再構成する」MIM と「表現を予測する」JEPA の対比
- [[concepts/denoising-autoencoder]] — 「破損 → 復元」枠組みの祖先（JEPA は復元先を表現空間に置いた変種）
- [[entities/mae]] — 構造が最も近い生成型の対照群
- [[entities/byol]] / [[entities/dino]] — EMA ターゲットによる崩壊回避という共通の仕掛け
