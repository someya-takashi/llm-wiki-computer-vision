---
type: entity
entity_kind: model
aliases: [I-JEPA, Image-JEPA, Image-based Joint-Embedding Predictive Architecture]
tags: [i-jepa, jepa, self-supervised-learning, vision-transformer, fair, lecun]
related: ["[[concepts/joint-embedding-predictive-architecture]]", "[[concepts/self-supervised-learning]]", "[[concepts/masked-image-modeling]]", "[[entities/mae]]", "[[entities/dino]]", "[[entities/ibot]]", "[[entities/byol]]"]
sources: ["[[sources/i-jepa]]"]
updated: 2026-06-17
---

# I-JEPA（Image-based Joint-Embedding Predictive Architecture）

Meta AI（FAIR）の **Mahmoud Assran, Yann LeCun ら**による画像自己教師あり学習モデル（arXiv:2301.08243, **CVPR 2023**）。詳細な解説: [[sources/i-jepa]]、パラダイム解説: [[concepts/joint-embedding-predictive-architecture]]。

## 一言で

**「1 つのコンテキストブロックから、同じ画像の複数ターゲットブロックの *表現* を予測する」** データ拡張なしの SSL。MAE（[[entities/mae]]）の画素再構成を表現空間予測に置き換えたもので、JEPA（[[concepts/joint-embedding-predictive-architecture]]）の最初の本格的画像実装。V-JEPA / V-JEPA 2 へ続く系譜の起点。

## アーキテクチャ

3 つの ViT（[[concepts/vision-transformer]]）で構成。

- **コンテキストエンコーダ** $f_\theta$: 標準 ViT。コンテキストブロックの **可視パッチのみ** 処理。
- **ターゲットエンコーダ** $f_{\bar\theta}$: 標準 ViT。画像全体を処理し、その **出力をマスク** してターゲット表現を作る。重みはコンテキストエンコーダの **EMA** で更新（momentum 0.996 → 1.0 へ線形増加）。
- **予測器** $g_\phi$: **狭い（narrow）ViT**。埋め込み次元を **384 に固定**（幅ボトルネック）、ヘッド数はバックボーンと同じ。深さ: ViT-B/16 で 6、ViT-L/16・ViT-H で 12、ViT-G/16 で 16。
- **[cls] トークンなし**。評価はターゲットエンコーダ出力の平均プーリング。
- **損失**: 予測表現とターゲット表現の平均 **L2 距離** のみ。

**マスキング（multi-block）**: ターゲット 4 個（scale 0.15–0.2, アスペクト比 0.75–1.5）+ コンテキスト 1 個（scale 0.85–1.0, アスペクト比 1）、コンテキストからターゲット重複領域を除去。

**最適化**: AdamW、バッチ 2048、lr 1e-4→1e-3（15 ep warmup）→ 1e-6（cosine）、weight decay 0.04→0.4 線形増加。

## 主要バリアント・結果

**表**: I-JEPA の主要構成と性能（[[sources/i-jepa]] より）

| バックボーン | 解像度 | 事前学習 | IN-1K linear | IN-1% | 備考 |
|---|---|---|---|---|---|
| ViT-B/16 | 224 | IN1k 600ep | 72.9 | — | 予測器深さ 6 |
| ViT-L/16 | 224 | IN1k 600ep | 77.5 | 69.4 | 予測器深さ 12 |
| ViT-H/14 | 224 | IN1k 300ep | 79.3 | 73.3 | < 1200 GPU 時間 |
| ViT-H/16₄₄₈ | 448 | IN1k 300ep | **81.1** | **77.3** | iBOT ViT-L に並ぶ |
| ViT-G/16 | — | IN22k | — | — | 意味的タスク最強・低レベルは伸びず |

- **凍結特徴量（線形プロービング）**: 拡張なしの MAE/CAE/data2vec を上回り、拡張ありの iBOT ViT-L（81.0）に匹敵。
- **低レベル/密予測（Clevr）**: 深度予測で DINO/iBOT を大差で上回る（Clevr/Dist 72.4 vs DINO 53.4）。
- **効率**: ViT-H/14 を 16× A100 で 72 時間未満。iBOT ViT-S/16 比 2.5× 速、MAE ViT-H/14 比 10× 効率。約 5× 少ない反復で収束。
- **フル IN ファインチューン**: ViT-H/16₄₄₈ で 87.1%（MAE 87.8 に 5.3× 少ないエポックで肉薄）。

## 系譜・位置づけ

- **構造の対照群は [[entities/mae]]**（画素再構成 vs 表現予測）。EMA ターゲットの仕掛けは [[entities/byol]] / [[entities/dino]] 由来。
- **最も近い先行研究は data2vec**（Meta のマルチモーダル自己蒸留 SSL、本 wiki 未取り込み）と Context Autoencoders（CAE）。
- **凍結特徴量の絶対性能では [[entities/dinov2]]（iBOT 路線のスケール版）に及ばない**が、計算効率・低レベルタスク汎用性・データ拡張不要という別軸で価値を主張。
- **後継**: V-JEPA（動画, 2024）→ V-JEPA 2（2025）。LeCun の世界モデル構想が画像 → 動画 → ロボット制御へ展開する起点。

## 公開

- コード/重み: Meta AI 公式（`github.com/facebookresearch/ijepa`）。

## 関連ページ

- [[sources/i-jepa]] — 論文要約（最重要）
- [[concepts/joint-embedding-predictive-architecture]] — JEPA パラダイム解説
- [[entities/mae]] — 構造が最も近い対照群
- [[entities/dino]] / [[entities/ibot]] / [[entities/dinov2]] — 比較対象の対比型・ハイブリッド系統
- [[entities/byol]] — EMA ターゲット + predictor の共通発想
