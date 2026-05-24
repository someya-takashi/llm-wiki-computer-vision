---
type: entity
entity_kind: model
aliases: [SigLIP, SigLIP 2, Sigmoid Loss Image-Language Pre-training]
tags: [weakly-supervised, vision-language, foundation-model, google]
related: [[concepts/weakly-supervised-pretraining]], [[concepts/foundation-model]], [[concepts/vision-transformer]]
sources: [[sources/dinov3]]
updated: 2026-05-24
---

# SigLIP / SigLIP 2

## 概要

**SigLIP** = **Sigmoid Loss Image-Language Pre-training**。Google Research の Zhai ら（2023, ICCV）が発表した CLIP 系の改良。CLIP の **softmax-based 対比損失を sigmoid loss に置き換え**、大規模バッチを必要とせず効率的に学習できるようにした。**SigLIP 2**（2024）はさらにスケールと改良を加えた後継。

- SigLIP 論文: "Sigmoid Loss for Language Image Pre-Training"（arXiv:2303.15343, ICCV 2023）
- SigLIP 2 論文: "SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features"（arXiv:2502.14786, 2025）
- コード: <https://github.com/google-research/big_vision>

DINOv3（[[entities/dinov3]]）論文において、PE（[[entities/perception-encoder]]）と並ぶ**主要な弱教師ありベースライン**として登場。

---

## CLIP との違い: sigmoid loss

### CLIP の softmax loss

CLIP（[[entities/clip]]）はバッチ内の N 個の画像-テキスト対について：

1. $N \times N$ の類似度行列を作る
2. **行方向（画像→テキスト）と列方向（テキスト→画像）で softmax を取り、対角を最大化**

これには問題がある：
- softmax の分母にバッチ内**全ての他のサンプル**との類似度が必要
- **バッチサイズを大きくしないと負例の多様性が確保できない**（CLIP は 32K まで使う）
- 大バッチ → 大量メモリ → GPU 並列が複雑

### SigLIP の sigmoid loss

SigLIP は **各画像-テキスト対を独立に二値分類問題**として扱う：

$$
\mathcal{L} = -\sum_{i,j} \log \frac{1}{1 + \exp(-z_{ij} \cdot t_{ij})}
$$

- $z_{ij}$: 画像 $i$ とテキスト $j$ の埋め込み類似度（cosine × temperature + bias）
- $t_{ij}$: ラベル（対応する対なら +1、それ以外は -1）

これにより：
- バッチ内の他サンプルとの正規化が不要 → **小バッチでも安定**
- **学習率・温度などのハイパーパラメータが鈍感**
- メモリ効率が良く、訓練を分散させやすい

> **補足: 効率の意味** — CLIP で 32K バッチを使うと、$32{,}000^2 \approx 10^9$ 個の類似度を一度に softmax 計算する必要がある。SigLIP の sigmoid なら全ペアを並列に独立処理でき、必要メモリは bandwidth-bound だけ。これで CLIP よりはるかに小さなハードウェアで巨大データを訓練できる。

---

## SigLIP（初版, 2023）

### 主要設定

- **訓練データ**: WebLI（Google の Web Light Image-text データ）
- **アーキテクチャ**: ViT-B/16, L/14, **SO400m**（So-large 400M params, パッチ 14）, G/16
- **損失**: sigmoid loss
- **言語**: 主に英語

### 主な貢献

- バッチサイズが小さくても性能が出る（ImageNet zero-shot で 4K バッチで CLIP 32K に匹敵）
- **SO400m**（一般 ViT サイズ命名から外れたカスタムサイズ）が ViT-L と ViT-H の間で良いトレードオフ

---

## SigLIP 2（2024）

複数の方向で大幅改善：

### 1. 多言語化
- 「Multilingual Vision-Language Encoders」がタイトルに入る通り、複数言語の WebLI で訓練
- 英語以外でのゼロショット性能を大幅改善

### 2. 改善されたセマンティック理解
- **NaFlex**: 可変解像度・可変アスペクト比のネイティブサポート
- **マスク予測損失追加**: テキストと画像両側でマスクされたトークンを予測する補助目的
- **訓練データの拡大**: 40B+ 画像-テキスト対

### 3. dense feature の改善
- SigLIP 2 は dense タスクも意識した設計
- ただし依然として DINOv3 や PEspatial には劣る

### 主要モデル（SigLIP 2）

| Model | Patch | Notes |
|---|---|---|
| ViT-B/16, B/32 | 16, 32 | 軽量 |
| ViT-L/16 | 16 | 標準 |
| **ViT-g/16** | 16 | 主力 |
| ViT-SO400m | 14 | 効率重視 |

DINOv3 比較表での「SigLIP 2 g/16」が主に登場する。

---

## DINOv3 ベンチマークでの SigLIP 2

### 強い領域
- **ImageNet 分類**: 89.1 linear（DINOv3 88.4 を僅差で上回る）
- **OOD 分類** ImageNet-R: 92.2（DINOv3 91.1 を上回る）
- **ゼロショット分類**: 83.1 IN1k（DINOv3 dino.txt 82.3 を上回る）
- **画像-テキスト検索**: COCO で I→T 71.4 / T→I 55.3

### 弱い領域
- **Dense linear probing**: ADE20k 42.7 vs DINOv3 55.9（大差）
- **3D correspondence**: NAVI 49.4 vs DINOv3 64.4
- **Video tracking**: DAVIS-L 62.9 vs DINOv3 83.3（大差）
- **Instance retrieval**: Oxford-H 32.7 vs DINOv3 60.7

DINOv3 著者が指摘する通り「**キャプションでは捉えきれないピクセルレベル情報は SigLIP でも掴めない**」という弱教師あり全般の限界が SigLIP にも当てはまる。

---

## SigLIP vs CLIP vs DINOv3

| 軸 | CLIP / OpenCLIP | SigLIP / SigLIP 2 | DINOv3 |
|---|---|---|---|
| 教師信号 | テキスト（softmax） | テキスト（sigmoid） | 画像のみ |
| バッチ要件 | 大（32K+）必須 | 小〜中で OK | 中〜大 |
| 効率 | 中 | **高** | 中 |
| ゼロショット | ◎ | ◎ | △（dino.txt 経由）|
| 画像-テキスト検索 | ◎ | ◎ | △ |
| Dense 予測 | △ | △ | **◎** |
| 細粒度分類 | ○ | ○ | **◎** |
| Instance 検索 | △ | △ | **◎** |
| 3D 認識 | △ | △ | **◎** |

各々得意分野が異なる。実用では**用途で使い分け**または併用する。

---

## 関連派生・後継

- **SigLIP 2 NaFlex**: 可変解像度・アスペクト比対応版
- **PaliGemma**: SigLIP + Gemma LLM（Google のオープン VLM）
- **PaLI-X / PaLI-3**: SigLIP を vision tower に持つ Google のマルチモーダルモデル

---

## なぜ重要か

1. **CLIP の限界を技術的に押し広げた**: 大バッチ要求を解消し、巨大データ訓練を民主化
2. **計算効率の良さで産業応用に強い**: PaliGemma など実用 VLM に組み込まれている
3. **多言語化により非英語圏での価値が高い**
4. **DINOv3 との比較ベンチマークでの「弱教師ありの代表」として**重要

---

## 関連ページ

- [[sources/dinov3]]: SigLIP 2 を主要ベースラインとして比較する論文
- [[entities/clip]]: SigLIP の元祖
- [[entities/perception-encoder]]: 同系統の Meta 競合
- [[entities/dinov3]]: 純粋 SSL 側の対抗馬
- [[concepts/weakly-supervised-pretraining]]: SigLIP が属するパラダイム
- [[concepts/foundation-model]]
