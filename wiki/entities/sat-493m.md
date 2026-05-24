---
type: entity
entity_kind: dataset
aliases: [SAT-493M, DINOv3 Satellite Dataset]
tags: [dataset, satellite, geospatial, ssl, meta-ai]
related: [[concepts/self-supervised-learning]], [[concepts/foundation-model]]
sources: [[sources/dinov3]]
updated: 2026-05-24
---

# SAT-493M

## 概要

**SAT-493M** = **Satellite-domain dataset, 4.93 億画像**。Meta AI が DINOv3 衛星版（[[entities/dinov3]] Sat 系列）の事前学習のために構築した、**Maxar の RGB ortho-rectified 衛星画像**から構築されたデータセット。

- 出典: [[sources/dinov3]] §8.1
- 画像数: **493M**（4.93 億枚）
- 画像サイズ: **512 × 512 ピクセル**
- 解像度（地上分解能）: **0.6 メートル/ピクセル**
- ソース: **Maxar Technologies**（商用衛星画像プロバイダ）
- 公開状況: **非公開**（Maxar 商用データのため再配布不可）

---

## なぜ重要か

LVD-1689M（[[entities/lvd-1689m]]）が自然画像で「SSL は基盤モデルになれる」を実証したのに対し、SAT-493M は「**SSL レシピが完全にドメイン非依存である**」ことを実証する。

- まったく**同じ DINOv3 のレシピ**（[[sources/dinov3]] §3-5）を、ハイパーパラメータをほぼ変えず（RGB 平均/std と訓練長だけ調整）に SAT-493M に適用
- 結果として **Geo-Bench 15 タスク中 12 で SOTA** を獲得し、**衛星専用に設計された Prithvi-v2 や DOFA を上回る**

これは「**ドメイン専用モデルを各分野で 1 から作る必要はない**」というメッセージを強く打ち出す結果で、医療画像・天文・産業画像など他分野への展開を後押しする。

---

## 構築手順

LVD-1689M とは異なり、**キュレーション工程は比較的シンプル**：

1. Maxar の RGB ortho-rectified imagery プールから
2. **ランダムサンプリングで 493M 枚**を抽出
3. 512 × 512 にクロップ

階層 k-means などの複雑なキュレーションは適用していない。これは：

- Maxar の画像自体がすでに地球表面を均等にカバーする商用データ
- 衛星画像はもともと「ジャンク」が少ない（人物自撮りやスクリーンショットなどがない）
- ドメインが狭いので Web 画像ほど偏向対策が必要ない

---

## 衛星画像ドメインの特性

自然画像（LVD-1689M）と何が違うか：

| | 自然画像 | 衛星画像（SAT-493M） |
|---|---|---|
| 視点 | 主に水平方向 | **真上から（nadir）** |
| スケール | 様々 | 一定（0.6m/pixel） |
| 対象 | 物体・人物・シーン | **建物・道路・植生・水域・農地** |
| 光源 | 様々 | 太陽（時間・季節・大気で変動） |
| センサー | 多種多様 | 衛星センサーの特性 |
| テクスチャ | 様々 | 細かい繰り返しパターンが多い |
| ラベル | 豊富（Web） | 希少・高価（注釈には専門知識が必要）|

SSL がこのような特殊ドメインで有用なのは、**「ラベルが希少」だがデータは豊富**という典型的状況だから。

---

## 訓練設定（衛星 DINOv3）

ウェブ版とほぼ同じ。違いは:

| 項目 | 値 |
|---|---|
| バックボーン | ViT-7B（teacher）, ViT-L（distilled） |
| 学習データ | SAT-493M |
| RGB 平均/std | 衛星画像用に再計算 |
| 初期事前訓練 | 100k iterations, 256² global crops |
| Gram anchoring フェーズ | 10k iterations |
| 高解像度ファインチューニング | 8k iterations, 512² |

その他（損失、optimizer、constant スケジュール、Gram anchoring の使い方）はすべてウェブ版と共通。

---

## 主要結果（§8）

### 樹冠高度推定（Open-Canopy）

- フランス全土 87,000 km² の SPOT 6-7 + LiDAR ground truth
- 4 チャネル（RGB + Infrared）入力に対応するため、patch embed 重みの 3 チャネル平均を 4 番目のチャネル重みとして追加

| Method | MAE↓ |
|---|---|
| [180] ViT-L | 2.42 |
| DINOv3 Web ViT-7B | 2.17 |
| DINOv3 Sat ViT-L | 2.07 |
| **DINOv3 Sat ViT-7B** | **2.02** |

### Geo-Bench（15 タスク平均）

| Method | Bands | Classification Mean | Segmentation Mean |
|---|---|---|---|
| DOFA ViT-L | all | 79.9 | 68.3 |
| Best Prithvi-v2 | all | 79.6 | 72.8 |
| [180] ViT-L | RGB | 77.8 | 72.9 |
| DINOv3 Sat ViT-L | RGB | 79.6 | 74.5 |
| DINOv3 Sat ViT-7B | RGB | 81.1 | 75.0 |
| **DINOv3 Web ViT-7B** | RGB | **81.6** | **75.9** |

**興味深い発見**: ウェブ版 DINOv3（LVD-1689M で学習）も多くの衛星タスクで競合的、特に semantic segmentation では衛星版を上回ることも。

> **補足: なぜウェブモデルが衛星タスクで効くのか** — 「物体の境界を正しく区切る」「テクスチャを認識する」「物体のスケール変化に対応する」などの能力は、ドメインが違っても転用可能。衛星専用学習が決定的に有利なのは、樹冠高度推定のような「**衛星センサー固有の物理的特性**（光の散乱、視差、量子化）に依存するメトリックタスク」のみ。

### 高解像度 semantic タスク（LoveDA, iSAID, DIOR）

| Method | LoveDA | iSAID | DIOR |
|---|---|---|---|
| Previous SOTA | 54.4 | 71.9 | 79.5 |
| DINOv3 Sat ViT-7B | 55.3 | 64.8 | 76.6 |
| **DINOv3 Web ViT-7B** | **56.2** | **71.4** | **80.5** |

LoveDA と DIOR では**ウェブ版**が SOTA を更新。

---

## ドメイン固有 vs 汎用の使い分け（§8.3 の結論）

著者らは「**タスク依存の使い分け**」を推奨：

| タスクタイプ | 推奨 |
|---|---|
| メトリックタスク（樹冠高度、深度推定） | **DINOv3 Sat**（衛星専用） |
| Semantic タスク（土地被覆分類、物体検出・セグメンテーション） | **DINOv3 Web** で十分、むしろ上回ることも |
| Multi-band 入力（Sentinel-2 の 10+ チャネル） | DINOv3 Web の patch embed 拡張 + 凍結で十分競合 |

---

## 注意点・限界

- **商用データ依存**: Maxar 画像は購入が必要。学術研究での再現は困難
- **RGB のみ**: マルチスペクトル（赤外、SAR、レーダー）は別途処理が必要
- **時系列なし**: 単一時刻の画像のみで学習。変化検出には別のアプローチが必要
- **解像度固定**: 0.6m/pixel に最適化されているため、他解像度のセンサー（Sentinel-2 の 10m/pixel など）への直接適用は性能低下しうる

---

## 関連ページ

- [[sources/dinov3]]: SAT-493M を構築・利用した論文（§8）
- [[entities/dinov3]]: SAT-493M で学習された衛星版モデル
- [[entities/lvd-1689m]]: 自然画像版の対応データセット
- [[concepts/self-supervised-learning]] / [[concepts/foundation-model]]
