---
type: source
source_path: raw/papers/SAM 2_ Segment Anything in Images and Videos.md
source_kind: paper
title: "SAM 2: Segment Anything in Images and Videos"
authors: [Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, Christoph Feichtenhofer]
year: 2024
venue: ICLR 2025
ingested: 2026-05-26
tags: [sam-2, segmentation, video, foundation-model, promptable, meta-fair, hiera, memory]
translation: [[translations/sam-2]]
---

# SAM 2: Segment Anything in Images and Videos

> 原典: [[translations/sam-2]] ・ `raw/papers/SAM 2_ Segment Anything in Images and Videos.md`
> 著者: Ravi, Gabeur, Hu, Hu, Ryali, Ma, Khedr, Rädle, Rolland, Gustafson, Mintun, Pan, Alwala, Carion, Wu, Girshick, Dollár, Feichtenhofer（Meta FAIR）
> 発表: arXiv:2408.00714（2024 年 8 月）→ ICLR 2025
> プロジェクト: <https://sam2.metademolab.com>

## 一言まとめ

**SAM（[[sources/segment-anything]]）の動画拡張版**。promptable segmentation を **PVS（Promptable Visual Segmentation, プロンプト可能視覚セグメンテーション）** タスクとして動画に一般化し、**streaming memory** を追加した Transformer 単一モデルで画像と動画を統一的に扱う（画像 = 1 フレーム動画）。同時にデータエンジンで **35.5M マスク × 50.9K 動画**の **SA-V** データセット（[[entities/sa-v]]、既存最大の VOS データセットの 53 倍）を構築・**CC by 4.0 で公開**。画像セグメンテーションでも SAM より **6 倍速 + 高精度**。

## 背景と問題意識

### SAM v1 の限界：画像のみ

[[entities/sam]]（2023）は画像セグメンテーション基盤モデルを確立したが、現実世界の視覚は時間次元を持つ：

- **動きと変形**: オブジェクトは動画内で外観・形状・姿勢が連続的に変化
- **遮蔽と再出現**: フレーム間で見えなくなり、また現れる（disappearance）
- **大量フレームの効率処理**: 動画は数百〜数千フレームに及び得る

既存の動画オブジェクトセグメンテーション（VOS）モデル（XMem, Cutie 等）は固定カテゴリに限定され、SAM のような「anything をセグメント」能力に達していなかった。

### 既存の "SAM + 動画トラッカー" の弱点

CLIP 時代の「SAM で 1 フレームをセグメント → 動画トラッカー（XMem 等）で伝播」アプローチには問題：

- **トラッカーがオブジェクト型に依存**: 任意のオブジェクトには弱い
- **SAM がビデオフレームで弱い**: 訓練が画像のみ
- **エラー修正が困難**: トラッカーがミスしたフレームを SAM で再アノテートし直し、トラッキングを再開する以外に方法がない

→ **モデルが過去フレームの予測を「記憶」しながら、任意のフレームでプロンプト修正可能** という設計が必要だった。

## 提案手法

### 1. PVS（Promptable Visual Segmentation）タスク

[[concepts/promptable-segmentation]] の動画版：

- **入力**: 動画の任意フレームで点・ボックス・マスクのプロンプト
- **出力**: 動画全体での masklet（時空間マスク、各フレームの 2D マスクの集合）
- **対話的修正**: 別フレームで追加プロンプトを与え、masklet を反復洗練
- **画像は 1 フレーム動画として扱う**: 統一モデル

> **補足: masklet とは何か** — VOS 系の用語で「**動画全体を通したオブジェクトの時空間マスク**」を意味する。各フレームでの 2D セグメンテーションマスクの時間方向の系列。SAM v1 が出力する単一マスクを、SAM 2 では masklet（動画全体）に拡張。

PVS は [[concepts/promptable-segmentation]] の自然な一般化で、半教師あり VOS（第 1 フレームのマスクのみ）と対話的 VOS（複数フレームのスクリブル）も特殊ケースとして包含する。

### 2. アーキテクチャ：SAM + streaming memory

<figure>

![](../../raw/assets/sam-2/fig3-architecture.png)

<figcaption>図3（再掲）: SAM 2 アーキテクチャ。フレームが 1 つずつ画像エンコーダで処理され、過去フレームのメモリにクロス注意される。マスクデコーダはオプションでプロンプトも取り、そのフレームのマスクを予測する。memory encoder が予測と画像埋め込みを将来フレーム用に変換する。</figcaption>
</figure>

| コンポーネント | 構成 | 役割 |
|---|---|---|
| **画像エンコーダ** | **Hiera**（hierarchical ViT, MAE 事前学習、[[entities/hiera]]） | 各フレームを 1 回だけエンコード（streaming）。SAM v1 の plain ViT より高速かつ多スケール特徴を出す |
| **Memory attention** | L=4 transformer block、self-attention + memory bank への cross-attention + MLP | 現在フレームの埋め込みを過去フレームのメモリに条件付ける。**2D-RoPE**（[[concepts/rotary-position-embeddings]]）を使用 |
| **Prompt encoder** | SAM v1 と同一（点・ボックス・マスク） | テキストプロンプトは未サポート（SAM v1 は §7.5 で proof-of-concept として実装したが SAM 2 では削除） |
| **Mask decoder** | SAM v1 + 階層エンコーダからのスキップ接続 + **occlusion 予測ヘッド** | 現在フレームでマスクを予測。occlusion head は「このフレームにオブジェクトが見えるか」を予測 |
| **Memory encoder** | 出力マスクを畳み込みでダウンサンプル → 画像埋め込みと要素ごと和 → 軽量畳み込みで融合 | 現在の予測を将来フレーム用のメモリに変換 |
| **Memory bank** | FIFO キュー（最近 N=6 フレーム + 最大 M プロンプトフレーム）+ **object pointers** | 過去の予測と高レベル意味情報を保存 |

### 3. Memory の 3 構造（最大の発明）

SAM 2 の中核は「**何を memory に保存し、どう attention するか**」の設計：

1. **空間メモリ**（spatial memory features）: 過去フレームの memory encoder 出力。FIFO で最大 N=6 フレーム
2. **プロンプトメモリ**: プロンプトされたフレームのメモリ。FIFO で最大 M フレーム保存（プロンプトされたフレームは特別扱い）
3. **Object pointers**: 各フレームのマスクデコーダの出力トークンを軽量ベクトルとして保存。**高レベル意味情報**を担う

memory attention は空間メモリと object pointers の両方にクロス注意する。空間メモリには **時間位置エンコーディング** が加わる（短期動きの表現）が、プロンプトメモリには加わらない（訓練と推論の時間範囲が異なるため）。

> **補足: なぜ object pointer が重要か** — アブレーション（表 12）で「object pointer なしでは SA-V val で 64.5、LVOSv2 で 67.0」だが、object pointer ありで「68.3 / 71.6」に改善。**空間メモリ（pixel-level）と object pointer（semantic-level）の二重表現** が長期トラッキング・オブジェクト再識別の鍵という発見。

### 4. データエンジン（3 phase で 8.4 倍高速化）

[[entities/sa-1b]]（SAM v1）と同じ「モデル支援でアノテーションスケール」パターンを動画に拡張：

| Phase | モデル・イン・ザ・ループ | フレーム/秒 | 高速化 | 収集 |
|---|---|---|---|---|
| **Phase 1: SAM per frame** | 各フレームを SAM で個別アノテート、時間伝播なし | 37.8 s/frame | 1× | 16K masklet / 1.4K 動画 |
| **Phase 2: SAM + SAM 2 Mask** | SAM で第 1 フレーム → SAM 2 Mask（マスクプロンプトのみ）で伝播 → 修正は SAM でゼロから | 7.4 s/frame | **5.1×** | 63.5K masklet |
| **Phase 3: SAM 2** | 完全機能版 SAM 2（点・マスク・メモリ）。修正クリックのみで反復 | **4.5 s/frame** | **8.4×** | 197.0K masklet |

最終 SA-V には自動 masklet（Phase 3 SAM 2 に 32×32 グリッド + ズームインクロップ）が **451.7K** 追加され、合計 **642.6K masklet**。

### 5. 訓練レシピ

- **事前学習**: SA-1B のみで画像セグメンテーション学習（90k iter、SAM v1 と類似）
- **完全訓練**: SA-V + Internal + SA-1B 10% + DAVIS/MOSE/YouTubeVOS のミックスで 300k iter
- **共同学習戦略**: 各イテレーションで画像か動画のバッチを確率的にサンプル（データソースサイズ比例）
- **対話シミュレーション**: 8 フレーム系列、最大 2 フレームをプロンプト/修正、初期プロンプト確率 50% マスク / 25% クリック / 25% ボックス
- **損失**: focal:dice:IoU(ℓ1):occlusion = **20:1:1:1**（SAM v1 は IoU に MSE、SAM 2 は ℓ1 + sigmoid に変更）

## 実験結果と知見

### 動画タスク

**Promptable video segmentation（PVS, 図 6）**:
- 9 ゼロショット動画データセットで SAM+XMem++ と SAM+Cutie を一貫して上回る
- **3 倍少ない対話で同等以上の精度** を達成

**半教師あり VOS（表 4, 7）**:
- 17 データセット平均で SAM+XMem++/Cutie を全プロンプト型で圧倒
  - 1 click: 56.7 (Cutie) → **64.3** (SAM 2)
  - 3 click: 70.1 → **73.2**
  - mask: 74.1 → **77.6**
- MOSE val: 71.7 (Cutie+) → **77.2** (SAM 2 Hiera-L)
- DAVIS 2017 val: 88.1 → **91.6**
- LVOS val: 66.0 → **76.1**
- SA-V val/test: 61.3/62.8 → **75.6/77.6**（先行手法に対し 14 ポイント差）

> **補足: SA-V でのギャップが特に大きい意味** — SA-V は「any」オブジェクトの open-world VOS ベンチマーク。先行手法は閉じた語彙（人/車/動物等）に最適化されており、part-level や open-world セグメンテーションでは大きく劣る。SAM 2 の +14 ポイントは「**動画基盤モデル**」と「**特化型 VOS モデル**」の境界が明確になったことを示す。

### 画像タスク（重要：SAM v1 を上回る）

37 データセットでの平均 1 クリック mIoU：

| モデル | データ | SA-23 All | FPS |
|---|---|---|---|
| **SAM ViT-H** | SA-1B | 58.1 | 21.7 |
| **SAM 2 (Hiera-B+)** | SA-1B のみ | **58.9** | **130.1**（6×） |
| **SAM 2 (Hiera-B+)** | our mix | **61.4** | **130.1** |

**SAM v1 と同じ SA-1B だけで訓練しても、Hiera 画像エンコーダのおかげで精度が上回り、しかも 6 倍速い**。動画データを追加すると 61.4% へさらに改善。

### スケーリング

- **データ量**: 図 7 で SA-V 量と精度の **冪乗則関係** を確認
- **モデルサイズ**: Hiera T → S → B+ → L で一貫した改善（特に動画メトリクス）
- **解像度**: 512 → 1024 で大きな改善

### 公平性（表 5, §6.1.3）

性別・年齢グループで概ね均等な性能。1 click 時のみ性別差が見えるが、これは「人物の代わりに部位（シャツ等）を予測する」プロンプト曖昧性に主に起因（manual inspection で確認）。3 click または mask プロンプトでは差消失。

## なぜ重要か

SAM 2 は CV foundation model の重要な拡張：

1. **動画への基盤モデル拡張**: CLIP / DINOv2 / SAM v1 は基本的に画像。**動画ドメインの foundation model** はまだ未開拓だった
2. **画像と動画の統一**: 画像を 1 フレーム動画として扱う設計で、SAM v1 を**置き換える上位互換**となった（画像でも 6× 高速）
3. **Streaming memory の設計パターン**: 「空間メモリ + object pointer」の二重表現は、後の動画 VLM（Gemini Vision、GPT-4V 動画モード）などにも影響
4. **SA-V の CC by 4.0 公開**: SA-1B（研究目的のみ）と異なり完全オープンライセンスで、**動画セグメンテーション分野の de facto 訓練/評価データ** に
5. **Hiera 画像エンコーダの実用性証明**: MAE pre-trained hierarchical ViT が plain ViT を実用面で凌駕することを示した

CV 基盤モデルの系譜：

| 系統 | 静的画像 | 動画 |
|---|---|---|
| **画像-テキスト WSL** | [[entities/clip]] | （未開拓） |
| **純粋 SSL** | [[entities/dinov2]] / [[entities/dinov3]] / [[entities/mae]] / [[entities/ibot]] | V-JEPA / V-JEPA 2（一部） |
| **プロンプト可能セグメンテーション** | [[entities/sam]] | **SAM 2**（[[entities/sam-2]]） |

## 限界（§B + コミュニティ知見）

1. **ショット変化を横断したオブジェクトのトラッキングに失敗** — フレーム境界でのコンテキスト切断
2. **長時間遮蔽や混雑シーンでオブジェクトを失う/混同**
3. **非常に細い・細かい詳細を持つ高速移動オブジェクトに苦戦**
4. **類似外観の近接オブジェクト**（複数の同一ジャグリングボール等）で混同
5. **オブジェクト間通信なし** — 各オブジェクトを独立に処理、共有オブジェクトレベルコンテキストを使わない
6. **テキストプロンプト未サポート**（SAM v1 §7.5 にあった proof-of-concept は削除）
7. **データエンジンは依然として人間アノテーターに依存** — 自動化の余地あり

## 用語と略称

- **SAM 2** = Segment Anything Model 2（[[entities/sam-2]]）
- **PVS** = Promptable Visual Segmentation（プロンプト可能視覚セグメンテーション）
- **SA-V** = Segment Anything Video dataset（[[entities/sa-v]]）
- **Hiera** = Hierarchical vision transformer（[[entities/hiera]]、SAM 2 の画像エンコーダ）
- **masklet** = 動画全体を通したオブジェクトの時空間マスク
- **streaming memory** = フレームを 1 つずつ処理しながら memory bank に蓄積する設計
- **memory attention** = 現在フレームを過去メモリに条件付ける attention 層
- **object pointer** = マスクデコーダ出力トークンから派生する高レベル意味情報ベクトル
- **occlusion prediction head** = フレームにオブジェクトが存在するかを予測するヘッド
- **VOS** = Video Object Segmentation
- **iVOS** = interactive VOS（対話的 VOS）
- **𝒥&ℱ** = VOS の標準精度指標（領域類似度 𝒥 = IoU と境界精度 ℱ の平均）
- **𝒢** = YTVOS 2019 の指標
- **disappearance rate** = アノテーション masklet のうち少なくとも 1 フレームで消失して再出現するものの割合
- **FPN** = Feature Pyramid Network（Hiera 出力を融合）
- **2D-RoPE** = 2 次元 Rotary Positional Embedding（memory attention に使用、[[concepts/rotary-position-embeddings]]）
- **RPB** = Relative Positional Bias（SAM v1 にあったが SAM 2 では削除）
- **FlashAttention** = 効率的注意カーネル実装
- **GRU** = Gated Recurrent Unit（memory に試したが効果なし）

## 関連ページ

- [[translations/sam-2]] — 全文和訳（Abstract + §1-9 + Appendix A-G）
- [[entities/sam-2]] — SAM 2 モデルファミリーのスペック
- [[entities/sa-v]] — SA-V データセットの詳細
- [[entities/hiera]] — 画像エンコーダ Hiera
- [[sources/segment-anything]] — SAM v1 の原典（PVS が拡張する元のタスク）
- [[entities/sam]] — SAM v1（SAM 2 の前身）
- [[entities/sa-1b]] — SAM v1 の訓練データ（SAM 2 でも 10〜15% 含まれる）
- [[concepts/promptable-segmentation]] — PVS が拡張する元のタスクパラダイム
- [[concepts/rotary-position-embeddings]] — 2D-RoPE が memory attention に使用
- [[entities/mae]] — Hiera が MAE で事前学習される
- [[concepts/foundation-model]] — SAM 2 を動画 foundation model として位置づけ
