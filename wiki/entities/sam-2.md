---
type: entity
entity_kind: model
aliases: [SAM 2, Segment Anything Model 2, SAM v2]
tags: [model, segmentation, video, foundation-model, meta-fair, promptable, hiera]
related: ["[[concepts/promptable-segmentation]]", "[[concepts/foundation-model]]", "[[concepts/rotary-position-embeddings]]"]
sources: ["[[sources/sam-2]]"]
updated: 2026-05-26
---

# SAM 2（Segment Anything Model 2）

## 概要

**SAM 2** = **Segment Anything Model 2**。Meta AI FAIR チーム（Ravi ら, 2024）が発表した、**画像と動画のための統一プロンプト可能セグメンテーション基盤モデル**。SAM v1（[[entities/sam]]）の動画拡張版で、画像を「1 フレーム動画」として扱うことで、SAM v1 を**置き換える上位互換**となっている（画像でも 6× 高速 + 高精度）。

- 論文: "SAM 2: Segment Anything in Images and Videos"
- arXiv: 2408.00714（2024 年 8 月）→ ICLR 2025
- コード: <https://github.com/facebookresearch/segment-anything-2>
- デモ: <https://sam2.metademolab.com>
- ライセンス: **Apache 2.0**（モデル）/ **CC by 4.0**（SA-V データセット、SAM v1 より寛容に）
- 詳細解説: [[sources/sam-2]] / 翻訳: [[translations/sam-2]]

## アーキテクチャ

SAM v1 の 3 コンポーネントに **streaming memory 系** が追加された 5 コンポーネント設計：

| コンポーネント | 構成 | SAM v1 との違い |
|---|---|---|
| **画像エンコーダ** | **Hiera**（hierarchical ViT, MAE 事前学習、[[entities/hiera]]）+ FPN | SAM v1 の plain ViT → **Hiera に変更**（多スケール特徴、6× 高速） |
| **Memory attention** | L=4 transformer block、self-attn + memory への cross-attn + MLP、**2D-RoPE** 使用 | **新規追加** |
| **Prompt encoder** | SAM v1 と同一（点・ボックス・マスク） | テキストプロンプトは削除 |
| **Mask decoder** | SAM v1 + 階層エンコーダのスキップ接続 + **occlusion 予測ヘッド** | occlusion head 追加（フレームにオブジェクトが見えるかを予測） |
| **Memory encoder** | 出力マスクを畳み込みで downsample → 画像埋め込みと要素和 → 軽量畳み込み融合 | **新規追加** |
| **Memory bank** | FIFO キュー: 最近 N=6 空間メモリ + M プロンプトメモリ + **object pointers** | **新規追加** |

**画像 1 枚あたり 1 回**の画像エンコーダ実行 + プロンプト → マスクで動画フレームをストリーミング処理。

## モデルファミリー

公開された SAM 2 は 4 種類の画像エンコーダ：

| 名前 | 画像エンコーダ | パラメータ数 | FPS（A100） | チェックポイント |
|---|---|---|---|---|
| **SAM 2 Hiera-T** | Hiera-T | 38.9M | 高速 | `sam2_hiera_tiny.pt` |
| **SAM 2 Hiera-S** | Hiera-S | 46M | 高速 | `sam2_hiera_small.pt` |
| **SAM 2 Hiera-B+** | Hiera-Base+ | 80.8M | **43.8**（動画 VOS） | `sam2_hiera_base_plus.pt` |
| **SAM 2 Hiera-L** | Hiera-Large | 224.4M | **30.2**（動画 VOS） | `sam2_hiera_large.pt` |

すべて 1024×1024 入力、Apache 2.0 で公開。

> **補足: Hiera-B+ がデフォルトの理由** — 速度と精度のバランス。画像 SA タスクで 130 FPS（SAM ViT-H の 6×）、動画 VOS で 43.8 FPS（実時間）。Hiera-L はさらに精度を上げるが 30 FPS。論文の主要結果は B+ 中心。

## 訓練レシピ

### 事前学習（画像のみ、SA-1B）

- データ: SA-1B のみ（[[entities/sa-1b]]）
- ステップ: 約 90k
- 解像度: 1024、bfloat16
- 最適化: AdamW（β₁,β₂=0.9, 0.999）、lr 4e-4、reciprocal sqrt スケジュール
- バッチサイズ: 256、画像あたり最大 64 マスク
- 修正クリック数: **7**（SAM v1 は 8、SAM v1 にあった "2 マスクのみイテレーション" は削除）
- 損失: focal:dice = 20:1 + **IoU の ℓ1 + sigmoid**（SAM v1 は MSE）
- 拡張: 水平反転 + 1024 正方形リサイズ

### 完全訓練（画像 + 動画）

- データ: **約 70% SA-V** + 約 14.8% Internal + 約 15.2% SA-1B（または + DAVIS/MOSE/YouTubeVOS で 49.5%/15.1%/15.5% に）
- ステップ: 約 300k、解像度 1024
- 学習率: **画像エンコーダ 6e-5 / その他 3e-4**（cosine スケジュール）
- バッチサイズ: 256
- **共同学習戦略**: 各イテレーションで画像か動画のバッチを確率的にサンプル（データソースサイズ比例）
- 動画拡張: hflip + affine + colorjitter + grayscale + per-frame colorjitter
- 損失: focal:dice:IoU(ℓ1):occlusion(CE) = **20:1:1:1**
- フレームあたり最大マスク: 画像 64 / 動画 3
- 8 フレーム系列、最大 2 フレームをプロンプト/修正

### 公開モデルの訓練計算量

- **256 A100 × 108 時間 ≈ 27,648 GPU 時間**
- 推定 12,165 kWh / **3.89 トン CO₂eq**（米国平均ガソリン車約 10,000 マイル走行に相当）

## 推論コード例

```python
from sam2.build_sam import build_sam2_video_predictor

# 動画予測器の初期化
predictor = build_sam2_video_predictor(
    config_file="sam2_hiera_l.yaml",
    ckpt_path="sam2_hiera_large.pt",
)

# 動画をエンコード（画像エンコーダは 1 回だけ走る）
inference_state = predictor.init_state(video_path="path/to/frames/")

# 第 1 フレームにクリックプロンプト
_, _, masks = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=0,
    obj_id=1,
    points=np.array([[500, 375]]),
    labels=np.array([1]),  # 1=foreground
)

# 動画全体に伝播
for frame_idx, obj_ids, masks in predictor.propagate_in_video(inference_state):
    # 各フレームのマスクが順次返ってくる
    process(frame_idx, masks)

# 後で別のフレームに修正クリック追加
predictor.add_new_points(inference_state, frame_idx=50, obj_id=1, ...)
```

画像用 SAM v1 互換 API もある：

```python
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2 = build_sam2(config_file="sam2_hiera_l.yaml", ckpt_path="...")
predictor = SAM2ImagePredictor(sam2)

predictor.set_image(image)
masks, scores, logits = predictor.predict(point_coords=..., point_labels=...)
```

## 主要結果

### 動画タスク

| ベンチマーク | 指標 | SAM 2 (Hiera-L) | 先行 SOTA |
|---|---|---|---|
| MOSE val | 𝒥&ℱ | **77.2** | Cutie+ 71.7 |
| DAVIS 2017 val | 𝒥&ℱ | **91.6** | JointFormer 90.1 |
| LVOS val | 𝒥&ℱ | **76.1** | Cutie 66.0 |
| **SA-V val** | 𝒥&ℱ | **75.6** | Cutie+ 61.3 |
| **SA-V test** | 𝒥&ℱ | **77.6** | Cutie+ 62.8 |
| YTVOS 2019 val | 𝒢 | **89.1** | Cutie+ 87.5 |

特に **SA-V では +14 ポイント** の大差。「any object」open-world VOS で先行を圧倒。

### 画像タスク（SAM v1 比較）

| モデル | データ | SA-23 1-click mIoU | FPS |
|---|---|---|---|
| SAM ViT-H | SA-1B | 58.1 | 21.7 |
| **SAM 2 Hiera-B+** | SA-1B のみ | **58.9** | **130.1**（6×） |
| **SAM 2 Hiera-B+** | our mix | **61.4** | 130.1 |
| **SAM 2 Hiera-L** | our mix | **63.5** | 61.4 |

**同じデータでも SAM v1 を上回る**（Hiera 画像エンコーダの効果）。さらに動画データを追加するとさらに改善。

## SAM v1 と SAM 2 の違いまとめ

| | SAM v1（2023） | SAM 2（2024） |
|---|---|---|
| **対象** | 画像のみ | 画像 + 動画（統一） |
| **画像エンコーダ** | ViT-B/L/H（plain） | **Hiera-T/S/B+/L**（hierarchical） |
| **メモリ** | なし | **空間 + プロンプト + object pointer** |
| **occlusion head** | なし | あり |
| **テキストプロンプト** | proof-of-concept（§7.5） | 削除 |
| **訓練データ** | SA-1B | **SA-1B + SA-V + Internal** |
| **画像 mIoU**（SA-23） | 58.1（ViT-H） | **63.5**（Hiera-L、our mix） |
| **画像 FPS** | 21.7（ViT-H） | **130.1**（Hiera-B+） |
| **データセットライセンス** | research only | **CC by 4.0** |
| **モデルライセンス** | Apache 2.0 | Apache 2.0 |
| **訓練 GPU 時間** | 256 A100 × 68 h（17,408 GPU·h） | 256 A100 × 108 h（27,648 GPU·h） |
| **CO₂ 排出** | 2.8 t | 3.89 t |
| **修正クリック数（事前学習）** | 8 | **7**（"2 マスクのみイテレーション" 削除） |
| **IoU 損失** | MSE | **ℓ1 + sigmoid**（より積極的） |

## 派生モデル

SAM v1 の派生（MobileSAM, HQ-SAM, EfficientSAM 等）と並んで、SAM 2 にも派生が急速に登場：

| モデル | 提案 | 特徴 |
|---|---|---|
| **SAM 2.1**（Meta, 2024 後半） | Meta | 訓練改善版、データミックス調整 |
| **EdgeSAM 2** | コミュニティ | モバイル/エッジ向け軽量版 |
| **Grounded-SAM 2** | IDEA Research | Grounding DINO + SAM 2 のテキスト → ボックス → マスクパイプライン |
| **SAM 2 medical** | 医療界コミュニティ | 医療画像/動画への適応 |
| **Track-Anything 2** | 各種 | SAM 2 をベースにしたインタラクティブ動画編集ツール |
| **SAM 3**（Meta Superintelligence Labs, 2025, [[entities/sam-3]] / [[sources/sam-3]]） | Carion et al. | **直接の後継**。PCS タスク導入、PE backbone に変更、DETR detector + presence head 追加。SAM 2 tracker は継承し PVS でも SAM 2 を凌駕（MOSEv2 で +12.4） |

## なぜ重要か

CV foundation model の動画ドメインへの拡張：

1. **動画基盤モデルの確立**: CLIP / DINOv2 / SAM v1 は基本的に画像のみ。SAM 2 は**動画ドメイン foundation model** の de facto 標準
2. **画像でも SAM v1 を凌駕**: Hiera の採用で画像セグメンテーションでも 6× 高速 + 高精度
3. **Streaming memory パターン**: 「空間メモリ + object pointer」の二重表現は、後続の動画 VLM（Gemini Vision、GPT-4V 動画モード）にも影響
4. **CC by 4.0 ライセンスのデータ**: SA-V の完全オープン公開で、動画セグメンテーション分野の de facto 訓練・評価データに

## 限界（§B + コミュニティ知見）

1. **ショット変化を横断するオブジェクトに失敗**
2. **長時間遮蔽・混雑シーンでオブジェクトを混同/失う**
3. **細い・細かい高速移動オブジェクトに苦戦**
4. **類似外観の近接オブジェクト**（複数同一物体）で混同
5. **オブジェクト間通信なし** — 各オブジェクトを独立処理
6. **テキストプロンプト未サポート**
7. **アノテーション自動化の余地**

## 関連ページ

- [[sources/sam-2]] — 原典の要約
- [[translations/sam-2]] — 全文和訳
- [[entities/sam-3]] — 直接の後継（PCS タスク追加、SAM 2 tracker を継承、PE backbone に変更）
- [[entities/sa-v]] — 訓練データセット
- [[entities/hiera]] — 画像エンコーダ
- [[entities/sam]] — SAM v1（前身）
- [[entities/sa-1b]] — SAM v1 訓練データ（SAM 2 でも 10-15% 含まれる）
- [[concepts/promptable-segmentation]] — タスクパラダイム（PVS = その動画版）
- [[concepts/rotary-position-embeddings]] — 2D-RoPE が memory attention に使用
- [[entities/mae]] — Hiera が MAE 事前学習される
- [[concepts/foundation-model]] — 動画 foundation model としての位置づけ
