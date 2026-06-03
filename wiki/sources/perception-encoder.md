---
type: source
source_path: raw/papers/Perception Encoder- The best visual embeddings are not at the output of the network.pdf
source_kind: paper
title: "Perception Encoder: The best visual embeddings are not at the output of the network"
authors: [Daniel Bolya, Po-Yao Huang, Peize Sun, Jang Hyun Cho, Andrea Madotto, Chen Wei, Tengyu Ma, Jiale Zhi, Jathushan Rajasegaran, Hanoona Rasheed, Junke Wang, Marco Monteiro, Hu Xu, Shiyu Dong, Nikhila Ravi, Daniel Li, Piotr Dollár, Christoph Feichtenhofer]
year: 2025
venue: NeurIPS 2025
ingested: 2026-05-28
tags: [perception-encoder, contrastive, vision-language, alignment-tuning, intermediate-layer-features, foundation-model, meta-ai, video-data-engine, sam-distillation]
translation: "[[translations/perception-encoder]]"
---

# Perception Encoder: 最良の視覚埋め込みはネットワークの出力にはない

> 原典: [[translations/perception-encoder]] ・ `raw/papers/Perception Encoder- The best visual embeddings are not at the output of the network.pdf`
> 著者: Daniel Bolya, Christoph Feichtenhofer ら（Meta / UT Austin / MBZUAI / Fudan University）
> 出典: NeurIPS 2025 / arXiv 2504.13181
> arXiv: <https://arxiv.org/abs/2504.13181>
> モデル・コード: <https://github.com/facebookresearch/perception_models>
> 動画データセット: <https://ai.meta.com/datasets/pe-video/>

## 一言まとめ

**「視覚-言語対比学習（CLIP 流）を頑健に・大規模にスケールすると、ネットワークの中間層に OCR / VQA / grounding / 検出 / 深度 / 追跡すべてに使える一般特徴量が自然に育つ。ただし最終層の出力には現れないので、`alignment tuning`（言語と空間の 2 種）で末端に引き出す」** という発見と、それを 1 つの統一エンコーダファミリー（PEcore / PElang / PEspatial）で具体化した論文。**画像・動画ゼロショット、MLLM、検出、深度、追跡で SOTA を同時に達成**。

## 背景と問題意識

[[entities/clip]]（2021）以降、視覚エンコーダの事前学習目的は分化してきた：

| 系統 | 代表モデル | 強い領域 | 弱い領域 |
|---|---|---|---|
| 視覚-言語対比学習（CLIP 流） | CLIP / [[entities/siglip]] / OpenCLIP | ゼロショット分類・検索 | dense 予測・OCR・空間性 |
| キャプション化（生成型） | AIMv2, CoCa | OCR / VQA / 言語 | 空間タスク |
| 空間的自己教師あり学習 | [[entities/dinov2]] / [[entities/dinov3]] | 検出・深度・追跡 | OCR / VQA |

問題は、**3 系統それぞれが「得意分野」を持ち、これらを統合するには技法を組み合わせざるをえない**こと。組み合わせ方は研究ごとにバラバラで、複雑さがスケーリングを困難にする。

> **補足: なぜ「単一の事前学習で全部対応」が望ましいか** — 訓練レシピが複雑になると、データ・損失関数・最適化器すべての組み合わせが指数的に増え、十億規模のデータと数千 GPU の計算予算では現実的に試せなくなる。**シンプルさは大規模化の必要条件**。

PE は「**対比学習だけでもスケールすれば全部できる**」というコントラリアン主張を実証する。

## 提案手法 / 主張

### 全体アーキテクチャ

PE は 3 つのモデルから成る：

```
                  ┌─────────────────────────┐
                  │   §2 PE_core            │  ← 1 つの大規模対比的事前学習
                  │   (G/14, 2B params)     │     画像 5.4B pairs + 22M videos
                  │   ゼロショット SOTA     │
                  └────────┬────────────────┘
                           │
                §3 中間層に一般特徴が育つことを発見
                           │
              §4 alignment tuning（言語）│ §5 alignment tuning（空間）
                           ↓                  ↓
                  ┌────────────────┐  ┌─────────────────┐
                  │   PE_lang G    │  │   PE_spatial G  │
                  │   MLLM 専門    │  │   dense 予測    │
                  └────────────────┘  └─────────────────┘
```

**鍵となる気づき**: PEcore の **中間層** にはタスクごとに最適な特徴量がすでに眠っている。それを alignment tuning で **最終層に引き出す** のがレシピの心臓部。

### 1. PEcore: 頑健な対比的事前学習（§2）

OpenCLIP ViT-L/14 をベースラインに、固定計算予算（1 ZFLOP）で 9 段階の累積アブレーション：

| # | 変更 | ImageNet val | 頑健性 avg | 重要度 |
|---|---|---|---|---|
| 1 | Baseline (OpenCLIP L/14) | 78.9 | 75.3 | — |
| 2 | Progressive resolution（98→154→224 を 4B ずつ）| 78.9 | 75.1 | ★★★（FLOPs 半減＆下流向上） |
| 3 | Batch size 32K → 64K | 79.5 | 76.2 | ★ |
| 4 | AdamW → LAMB | 79.9 | 76.9 | ★★ |
| 5 | High resolution 336 を追加段階 | 80.4 | 78.3 | ★★ |
| 6 | 2D RoPE 追加（学習位置埋め込みは保持） | 80.7 | 79.2 | ★★ |
| 7 | Attention pooling block（class token を入力に） | 81.0 | 80.1 | ★★ |
| 8 | Data augmentation（クロップ・色ジッター・反転） | 81.1 | 80.8 | ★ |
| 9 | Mask regularization（1/16 のバッチをマスク + cosine alignment） | 81.3 | 80.9 | ★ |

合計 ImageNet +2.4%、頑健性 +5.6%（同 FLOPs）。

> **補足: progressive resolution の意外な威力** — 98→154→224 解像度を段階的に上げる単純な工夫が、頑健性とスケーラビリティに大きく効く。論文 §3 の Fig. 5 では、これが下流タスク（COCO 検出）で **+10 mAP** という最大の効果を出している。グローバル token への過適合を防ぐ役割と推測。

**最大スケール（G/14）**: vision tower 1.88B + text tower 0.47B、解像度 448、5.4B unique image-text pairs を 86B samples seen まで訓練、バッチ 131K。

### 2. 動画データエンジン（§2.2）

ウェブ動画にはアラインされたキャプションが少ない問題への対処：

1. **Captioner 構築**: 早期 PE（フレーム単位）+ Llama を組み合わせて動画キャプション化モデルを訓練
2. **人手精錬**: 26.5 万動画を人間レーターでキャプション精錬 → 動画キャプション化器を再ファインチューン
3. **合成集約**: PE 動画キャプション + Llama 3.2 のフレーム単位キャプション + メタデータ（タイトル / 説明） → Llama 3.3 70B で「アラインされた」短いキャプションに統合

これで 22M 動画に高品質キャプションを付与し、PEcore をフレーム単位エンコーダとして 8 フレーム平均プーリングで動画ファインチューン。

> **重要な観察**: 動画ファインチューンは画像性能 *も* 大きく改善する（一般分類 +0.6%、細粒度 +1.2%、検索 +4.0%）。「動画を学ぶと画像が良くなる」現象は、動画キャプションの記述密度が高いため。

### 3. 中間層の一般特徴量の発見（§3）

PEcore G を AIMv2-3B（キャプション化）と DINOv2-g（SSL）と比較する **層ごとの凍結特徴解析**（図 4）：

- AIMv2 → **言語タスクで強いが空間タスクで弱い**
- DINOv2 → **空間タスクで強いが言語タスクで弱い**
- **PEcore G → 両方の中間層が AIMv2 や DINOv2 と並ぶか上回る**

これは **対比的損失がグローバル整合性しか要求しないにもかかわらず**、十分にスケールすれば豊かな汎用特徴量が中間層に育つことを意味する。最終層は CLIP loss を fit する過程で「圧縮されすぎて」一般情報を出力できなくなっている。

**スケーリング挙動（図 7）**:
- Vanilla CLIP レシピ: L スケール（300M）でプラトー
- 頑健なレシピ: G スケール（2B）まで *最良層の* 性能はスケール継続
- *最後の層* は依然として停滞 → **CLIP loss が一般特徴を「不明瞭化」** している証拠

> **CV 分野的意義**: これは「CLIP モデルは下流タスクにスケールしない」と主張した [^30] への直接的な反証であり、**対比学習の単純さを保ったまま、複雑な多目的事前学習を不要にする** 可能性を示す。

### 4. PElang: 言語アラインメント（§4）

「中間層に良い特徴がある」ことを実用化するため、`alignment tuning` を導入：

- PEcore G を Llama3.2 3B decoder にアラインメント
- 両方を **unfrozen** で 2 層 MLP 接続
- 最後の 3 層を破棄（[Chen et al., 2024]）
- LayerScale + DropPath で正則化
- OCR Q&A / キャプション化 / 視覚 Q&A / 動画 Q&A 全体で **70M サンプル**（事前学習比で約 1000 分の 1）
- 最終的に **視覚エンコーダのみを抽出** → PElang

驚くべきことに、**訓練ミックスに grounding データが含まれていなくても** RefCOCO（grounding）性能が大幅向上。これは PEcore G の中間層 grounding 特徴量が末端に「持ち上げられた」ため。

### 5. PEspatial: 空間アラインメント（§5）

dense タスクには 2 つの目標：
1. 層 40 付近の高レベル空間特徴（検出・深度向け）を維持
2. 層 33 以降で劣化する **locality**（局所性, 追跡向け）を回復

→ **2 教師アラインメント**：
- **教師 1**: PEcore G 自身の **凍結された層 41 特徴** → 自己蒸留で末端まで特徴を維持
- **教師 2**: **SAM 2.1 の mask logits**（特徴量ではなくマスク確率）を 32×32 グリッドで連結したもの → 局所性を強制

[[concepts/gram-anchoring]]（[[entities/dinov3]] が使う、Gram 行列で特徴関係を保つ手法）とは設計思想が異なるが、目的は近い：「過去の良い特徴量を後段で失わないようにする」。

> **DINOv3 との対比**: DINOv3 は「PE は dense で弱い」と主張していたが、本論文の PEspatial G は SAM 2.1 蒸留 + 自己蒸留で **dense 予測 SOTA**（ADE20k 49.3、NYU depth 0.262 RMSE）を達成。「弱教師あり vs 自己教師あり」の競争に巻き戻しが入った。

## 実験結果と知見

### ゼロショット画像（表 3）

ImageNet val、ロバストネス（6 平均）、細粒度分類（6 平均）、検索（COCO/Flickr 4 平均）すべてで PEcore が SOTA。**3 年ぶりに JFT-3B や WebLI なしで対比 SOTA を取り戻した**。

| Model | Encoder Params | Data | Avg Class. | Avg Fine. | Avg Retrieval |
|---|---|---|---|---|---|
| SigLIP2-g-opt | 1.1B | 10B | 86.2 | 77.4 | 76.3 |
| PEcore G (image only) | 1.9B | 5.4B | 86.0 | 76.1 | 78.6 |
| **PEcore G** | 1.9B | 5.4B + 22M videos | **86.6** | 79.4 | **78.9** |

### ゼロショット動画（表 4）

InternVideo2（フル時間 attention）にゼロショット分類で *勝ち*、検索でほぼ並ぶ。**シンプルなフレームレベルエンコーダ（8 フレーム平均）で、わずか 22M 動画**。

### MLLM ベンチマーク（表 5）

Llama 3.1 8B decoder 上で、PElang G は OCR / Chart / Doc Q&A 72.4、Visual Q&A 78.1、Captioning 120.1（CIDEr）、Grounding 71.3、Video 60.1 で **すべてのスケール・解像度・タスクで他のエンコーダを上回る**。

### Dense 予測（表 6, 7, 8）

- **追跡（DAVIS）**: PEspatial G 61.5 vs DINOv2-g 58.5
- **意味セグメンテーション（ADE20k）**: PEspatial G 49.3 vs DINOv2-g 48.7
- **深度（NYU RMSE↓）**: PEspatial G 0.262 vs DINOv2-g 0.279
- **物体検出（LVIS box AP）**: PEspatial G 54.2 vs DINOv2-g 51.5
- **COCO 絶対 SOTA**（DETA decoder + Objects365 追加）: PEspatial G **66.0 box AP**（CoDETR EVA02-L 65.9 を上回る）

**シンプルな DETR-style decoder（[[sources/detr]] / [[entities/detr]]）で複雑な検出専用モデルに並ぶ初の汎用対比モデル**。

## 限界・批判的視点

- **未公開データへの依存**: 5.4B image-text pairs は MetaCLIP キュレーションだが詳細データセット自体は非公開。再現性に制限。
- **3 モデル運用のコスト**: PEcore / PElang / PEspatial の 3 つを使い分ける必要があり、デプロイ上は CLIP より複雑。
- **video benchmark での僅差**: InternVideo2 が ActivityNet 検索で依然優位。動画ネイティブ訓練の効果は完全には吸収できていない。
- **SAM 2.1 への依存**: PEspatial が SAM 2.1 のマスクロジット教師に依拠している点は、「SAM のような supervised teacher なしでは dense 予測が解けないのか？」という DINOv3 の問いに完全な答えではない。
- **「中間層特徴量」の理論的解釈の欠如**: なぜ対比学習で中間層に多目的特徴が育つかは経験的観察のみで、理論的説明はない（Appendix C.5 の解析にとどまる）。
- **Appendix への分散**: 訓練詳細・ablation の多くが Appendix B-E に詳述されており、本文だけでは再現困難。

## 既存 wiki との接続

### CLIP 系統の正統な後継

[[sources/clip]] → [[sources/siglip]] → [[sources/siglip-2]] → **PE** という流れの中で、PE は **「データ規模とレシピ精錬」** に注力した立場。SigLIP 2 が **「全部入りレシピ」**（[[concepts/contrastive-learning]] + [[concepts/masked-image-modeling]] + 自己蒸留 + LocCa decoder + 多言語）で多目的化したのと対照的に、PE は **対比学習をピュアに保ったまま** 中間層の発見で同様の汎用性を実現する。

### DINOv3 との理論的対立軸

[[sources/dinov3]] は「PE のような WSL（弱教師あり学習, [[concepts/weakly-supervised-pretraining]]）モデルは dense feature と instance retrieval で SSL に勝てない」と主張していた。本論文の PEspatial はこの「dense で弱い」評価を **SAM 2.1 蒸留 + 自己蒸留** で覆す試み。両者は同じ Meta 内部の競合プロジェクトとして、**WSL vs SSL** という根源的問いへの異なる答えを提示する。

### SAM 3 の backbone

[[sources/sam-3]] / [[entities/sam-3]] は **PE を共有 backbone として採用**。PCS（[[concepts/promptable-concept-segmentation]]）タスクで PE の言語アラインメント能力が本領発揮。SAM 2 が PEspatial の「教師」役で、SAM 3 が PE の「弟子」役という、循環的な関係が成立している。

### 新概念ページ: alignment tuning

PE が導入した **alignment tuning**（[[concepts/alignment-tuning]]）は、CLIP ファミリーで以前から知られていた「中間層の方が良い」現象を、初めて系統的に活用したファインチューニング戦略。今後の foundation model 訓練レシピで標準パターンになる可能性。

## 用語と略称

- **PE** = Perception Encoder（Meta の視覚エンコーダファミリー）
- **PE_core / PE_lang / PE_spatial** = それぞれ汎用 / MLLM 用 / dense 用の 3 バリアント。**core が事前学習、lang と spatial は alignment tuning 後の派生**
- **alignment tuning**（アラインメント・チューニング） = 中間層の良い特徴量を最終層に「引き出す」ためのファインチューニング戦略。言語側は MLLM 統合、空間側は SAM 教師＋自己蒸留
- **CLIP** = Contrastive Language-Image Pre-training（[[entities/clip]]）
- **MLLM** = Multimodal Large Language Model（マルチモーダル大規模言語モデル）。テキストだけの LLM に視覚エンコーダを差し込んで画像入力に対応させた拡張
- **MetaCLIP** = Meta が CLIP の事前学習データキュレーション手法を再現・改善したもの（[Xu et al., 2024]）
- **LAMB** = Layer-wise Adaptive Moments optimizer for Batch training。大バッチ訓練向け最適化器。**EVA-CLIP も使用**
- **RoPE / 2D RoPE** = Rotary Position Embedding。回転ベースの位置符号化、可変解像度に強い。詳細は [[concepts/rotary-position-embeddings]]
- **Attention pooling** = CLIP 埋め込みを構築するための pool 層を、cross-attention transformer ブロックに置き換えた設計。class token を入力に保つことが小型モデル性能のカギ
- **Progressive resolution**（漸進的解像度） = 訓練中に画像解像度を段階的に上げる戦略（98 → 154 → 224 → 336 → 448）。**FLOPs を半減しつつ汎化性能向上**
- **Mask regularization** = 入力バッチの 1/16 を複製＆マスクし、出力でマスク部とマスクなし部の cosine 類似度を最大化する正則化
- **PVD** = PE Video Dataset。1M 動画 + 120K 人手精錬キャプション + 15K PVD Benchmark（詳細は Appendix A.1, 本文では概要のみ）
- **frozen features** = 凍結特徴量（バックボーンを学習せず特徴量だけを抽出して下流タスクに使う評価方法）
- **layer-wise feature analysis** = 各層の特徴量を独立に評価する解析手法。PE の「中間層の方が良い」発見の中核
- **ViTDet** = Vision Transformer Detector。検出向け ViT バックボーン構造
- **DETA** = Detection Transformer with Assignment（[[entities/detr]] の改善版、1 対 1 → 1 対多マッチングで収束加速、PEspatial の最終検出ベンチマークで使用）
- **CoDETR** = Collaborative-DETR（[Zong et al., 2023]）
- **JFT-3B / WebLI** = Google 内部の超巨大データセット（PE は **これらを使わずに対比 SOTA**）
- **MetaCLIP-G / SigLIP2-g-opt / DFN-H+ / EVA-CLIP-18B / InternVL-C** = 比較対象の対比モデル群
- **AIMv2** = Autoregressive Image Models v2（Apple のキャプション化型事前学習）
- **InternVideo2** = 動画ネイティブ事前学習モデル（102M 動画で訓練、PE の主要動画ベースライン）
- **SAM 2.1 mask logits** = SAM 2 のセグメンテーションマスク確率（特徴量ではなく出力）。PEspatial の教師として 32×32 グリッドで連結使用
- **layer 41** = PEcore G（50 層）の中間層番号。空間タスクで最も良好な性能を出す層
- **Gram anchoring** = [[concepts/gram-anchoring]]（DINOv3 が使う、Gram 行列で特徴関係を保つ正則化）。PEspatial の自己蒸留と目的は近いが手法は異なる

## 関連ページ

- [[translations/perception-encoder]] — 本文全文の和訳
- [[entities/perception-encoder]] — PE の詳細スペック（モデル変種・パラメータ・主要結果）
- [[concepts/alignment-tuning]] — PE が導入した中間層引き出しの一般戦略
- [[sources/dinov3]] — PE を主要比較相手として位置付ける対抗論文
- [[sources/sam-3]] — PE を backbone として採用
- [[entities/clip]] / [[sources/clip]] — PE の理論的祖先
- [[entities/siglip]] / [[sources/siglip-2]] — 並列に発展した「全部入りレシピ」型 WSL
- [[concepts/weakly-supervised-pretraining]] — PE が属するパラダイム
- [[concepts/contrastive-learning]] — PEcore の事前学習目的
- [[concepts/foundation-model]] — PE は CV foundation model の最新世代
