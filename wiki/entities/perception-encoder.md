---
type: entity
entity_kind: model
aliases: [Perception Encoder, PE, PEcore, PElang, PEspatial, PE-Core, PE-Lang, PE-Spatial]
tags: [weakly-supervised, vision-language, foundation-model, meta-ai, distillation, alignment-tuning]
related: ["[[concepts/weakly-supervised-pretraining]]", "[[concepts/foundation-model]]", "[[concepts/vision-transformer]]", "[[concepts/promptable-concept-segmentation]]", "[[concepts/alignment-tuning]]"]
sources: ["[[sources/perception-encoder]]", "[[sources/dinov3]]", "[[sources/sam-3]]", "[[sources/siglip-2]]"]
updated: 2026-05-28
---

# Perception Encoder（PE）

## 概要

**Perception Encoder（PE）** は Meta AI Research（Bolya, Feichtenhofer ら）が NeurIPS 2025 で発表した視覚基盤モデルファミリー。**画像-テキスト対比学習（CLIP スタイル）を頑健にスケールするだけで、ネットワークの中間層に多目的な一般特徴量が自然発生する** ことを発見し、それを `alignment tuning`（[[concepts/alignment-tuning]]）で末端に引き出して MLLM・dense 予測でも SOTA を達成する 3 バリアント構成。

- 論文: "Perception Encoder: The best visual embeddings are not at the output of the network"
- arXiv: 2504.13181 / NeurIPS 2025
- コード・モデル: <https://github.com/facebookresearch/perception_models>
- 動画データセット: <https://ai.meta.com/datasets/pe-video/>
- メーカー: Meta / UT Austin / MBZUAI / Fudan University

PE は **3 つのバリアント** で構成される：

| バリアント | 用途 | 派生方法 |
|---|---|---|
| **PEcore** | 汎用（ゼロショット分類・検索の SOTA） | 大規模対比的事前学習＋動画ファインチューン（事前学習本体） |
| **PElang** | MLLM 統合 | PEcore に **言語アラインメント**（Llama3.2 decoder と unfrozen 共同訓練） |
| **PEspatial** | dense 予測（検出・セグメンテーション・追跡・深度） | PEcore に **空間アラインメント**（SAM 2.1 mask logits + 自己層 41 特徴の 2 教師蒸留） |

3 バリアントすべてが **同じ PEcore 事前学習を起点**として、**短いアラインメントチューニング**（事前学習比で約 1% の追加サンプル）で得られる点が中核設計。

---

## PEcore: 大規模対比的事前学習

### 訓練データ（**訂正**）

| 量 | 意味 |
|---|---|
| **5.4B** | unique image-text pairs（MetaCLIP [Xu et al., 2024] でキュレーション） |
| **86B** | samples seen（G スケール）— 5.4B × 約 16 エポック相当 |
| 58B | samples seen（B / L スケール） |
| グローバルバッチ | 131K |
| 進行解像度 | 98 → 154 → 224 → 336 → 448（モデルに応じて段階数を変更） |
| 動画ファインチューン | **22M** 動画（独自データエンジンで再キャプション化、8 フレーム平均プーリング） |

> **注**: 以前の wiki エントリでは「86B image-text pairs」と書かれていたが、これは誤り。**5.4B unique pairs** を **86B samples 見るまで** 訓練するのが正確。

### モデル仕様（表 2）

| Scale | Vision Params | Vision Width × Depth | MLP | Heads | Text Params | CLIP Dim |
|---|---|---|---|---|---|---|
| **B** | 0.09B | 768 × 12 | 3072 | 12 | 0.31B | 1024 |
| **L** | 0.32B | 1024 × 24 | 4096 | 16 | 0.31B | 1024 |
| **G** | **1.88B** | **1536 × 50** | 8960 | 16 | 0.47B | **1280** |

主力は **G スケール**（vision 1.88B + text 0.47B = 計 2.35B）。

### 頑健な事前学習レシピ（§2.1 の 9 段階）

OpenCLIP ViT-L/14 をベースラインに、**FLOPs を増やさずに** 段階的に改善：

1. Progressive resolution（98→154→224 を 4B サンプルずつ）→ FLOPs 半減
2. Batch size 32K → 64K（hard negative 増加）
3. AdamW → **LAMB**（学習率を 5×10⁻⁴ → 2×10⁻³ に上げられる）
4. **336 解像度の追加段階**
5. **2D RoPE**（学習位置埋め込みも保持）
6. **Attention pooling block**（class token を入力に残すのが小型モデルでカギ）
7. データ拡張（heavy crop, brightness/saturation jitter, horizontal flip）
8. **Mask regularization**（1/16 バッチをマスク → cosine 整合）

累積効果: ImageNet val +2.4%、頑健性 avg-of-6 +5.6%、**COCO 検出（凍結特徴）+10 mAP**。

### 動画データエンジン（§2.2）

ウェブ動画に高品質キャプションが少ない問題への解決策：

1. 早期 PE をフレーム単位エンコーダ + Llama を decoder として動画キャプション化器を構築（PLM レシピ）
2. 26.5 万動画を人手精錬 → キャプション化器を再ファインチューン
3. 動画キャプション + フレーム単位キャプション（Llama 3.2）+ メタデータ（タイトル / 説明） → Llama 3.3 70B で短いアラインキャプションに統合

**動画ファインチューンは画像性能も改善**（一般分類 +0.6%、細粒度 +1.2%、検索 +4.0%）。

### PE Video Dataset（PVD）

公開動画データセット（詳細は本論文 Appendix A.1）：

- **1M 動画**: タグ・説明付き、動きを中心とした多様シーン
- **120K サブセット**: 人手精錬キャプション（平均長 57.1 words / Human-verified, 111.7 words / Long-automated）
- **PVD Benchmark**: 15K サンプルで構成、SigLIP / SigLIP2 / InternVL / PE を評価

### 主要結果

#### ゼロショット画像（表 3）

| Model | Params | Data | Avg Class. | Avg Fine. | Avg Retrieval |
|---|---|---|---|---|---|
| SigLIP2-B/16† | 0.1B | 10B | 71.1 | 68.5 | 69.0 |
| **PEcore B** | 0.1B | 5.4B | **73.2** | 66.1 | **72.7** |
| SigLIP2-L/16† | 0.3B | 10B | 83.3 | 75.5 | 75.6 |
| **PEcore L** | 0.3B | 5.4B | **83.9** | 74.6 | **75.7** |
| SigLIP2-g-opt | 1.1B | 10B | 86.2 | 77.4 | 76.3 |
| **PEcore G** | **1.9B** | 5.4B + 22M | **86.6** | **79.4** | **78.9** |

**3 年ぶりに JFT-3B / WebLI なしで対比 SOTA**。

#### ゼロショット動画（表 4）

| Model | Params | Video Data | Avg Class. | Avg Retrieval |
|---|---|---|---|---|
| InternVideo2 | 1.0B | **102M** | 70.7 | **59.9** |
| SigLIP2-g-opt | 1.1B | n/a | 68.2 | 46.6 |
| **PEcore G** | 1.9B | **22M** | **74.8** | 58.7 |

InternVideo2（native 動画モデル）に分類で勝利、検索で僅差。**シンプルなフレーム単位 8 フレーム平均で。**

---

## §3: 中間層に育つ一般特徴量の発見

### Layerwise Feature Analysis（図 4）

PEcore G、AIMv2-3B（キャプション化）、DINOv2-g（SSL）を 7 タスクで層ごとに比較：

| タスク | AIMv2 最良 | DINOv2 最良 | PEcore G 最良 |
|---|---|---|---|
| ImageNet 分類 | 89.5（層 24） | 86.9（層 38） | **89.8（層 50）** |
| OCR Q&A | **61.4（層 20）** | 38.9（層 23） | 62.1（層 38） |
| Visual Q&A | 77.5（層 24） | 63.7（層 27） | **75.3（層 47）** |
| Grounding RefCOCO | 67.9 | 70.6 | 68.7（層 38） |
| Detection COCO | 39.1 | 42.9 | **43.2（層 38）** |
| Depth NYU (↓) | 0.31 | 0.28 | **0.25（層 39）** |
| Tracking DAVIS | 54.7 | 58.5 | 55.8（層 32） |

**PEcore G は対比的損失だけで訓練されたにもかかわらず、AIMv2 の言語タスクと DINOv2 の空間タスクに匹敵 or 超える層を中間に持つ**。

### スケーリング挙動（図 7）

- vanilla CLIP: L スケール（300M）でプラトー
- 頑健レシピ: **G スケール（2B）まで *最良層* がスケール継続**
- **最後の層** は両方とも依然停滞 → CLIP loss が一般特徴を「不明瞭化」している証拠

---

## PElang: 言語アラインメント（§4）

### レシピ

| 要素 | 内容 |
|---|---|
| Decoder | **Llama3.2 3B**（テキストのみ事前学習済み） |
| 接続 | 2 層 MLP |
| Encoder/Decoder 状態 | **両方 unfrozen** |
| 削除 | PEcore の最後の 3 層を破棄 |
| 正則化 | LayerScale + DropPath |
| 訓練データ | OCR Q&A / Captioning / Visual Q&A / Video Q&A、計 **70M サンプル** |
| 損失 | 次トークン予測 |
| 最終出力 | 視覚エンコーダのみ抽出 → **PElang** |

### 主要結果（表 5、Llama 3.1 8B decoder で評価）

| Model | Avg OCR | Avg VQA | Avg Captioning | Avg Grounding | Avg Video |
|---|---|---|---|---|---|
| SigLIP2-so400M | 60.9 | 76.8 | 116.5 | 67.4 | 54.5 |
| PEcore G | 60.9 | 73.5 | 115.0 | 65.5 | 51.8 |
| **PElang G** | **72.4** | **78.1** | **120.1** | **71.3** | **60.1** |

**訓練ミックスに grounding が含まれないのに RefCOCO が +5pt 以上向上** — PEcore G 中間層の grounding 能力が末端に持ち上げられた証拠。

---

## PEspatial: 空間アラインメント（§5）

### 2 教師の組み合わせ

| 教師 | 目的 | 与える信号 |
|---|---|---|
| **PEcore G 自身の凍結層 41 特徴** | 高レベル空間意味の維持 | 自己蒸留 |
| **SAM 2.1 mask logits** | 局所性（locality）の回復 | 32×32 グリッドで連結したマスク確率 |

**SAM 単独だと意味論を失い、自己蒸留単独だと局所性が不足**。両方の組み合わせで両得。

### 重い正則化

- DropPath
- LayerScale
- **75% パッチマスキング**（[[concepts/masked-image-modeling]] 流）

### 主要結果

#### Frozen Dense Prediction（表 6, 448 解像度凍結）

| Encoder | DAVIS Best | ADE20k Best | NYU Best (↓) |
|---|---|---|---|
| DINOv2-g | 58.5 | 48.7 | 0.279 |
| PEcore G | 56.8 | 41.5 | 0.247 |
| **PEspatial G** | **61.5** | **49.3** | **0.262** |

#### End-to-End Detection（表 7, Mask R-CNN + ViTDet, 1024 解像度）

| Encoder | LVIS AP_box | LVIS AP_mask | COCO AP_box | COCO AP_mask |
|---|---|---|---|---|
| DINOv2-g | 51.5 | 47.3 | 57.2 | 50.0 |
| SigLIP2-g-opt | 52.9 | 48.5 | 57.1 | 50.2 |
| **PEspatial G** | **54.2** | **49.3** | **57.8** | **50.3** |

#### SOTA Setting（表 8, COCO val2017）

| Encoder | Params | Detector | COCO AP_box |
|---|---|---|---|
| SwinV2-G | 3.0B | HTC++ | 62.5 |
| InternImage-G | 3.0B | DINO | 65.3 |
| EVA02-L | 0.3B | CoDETR | 65.9 |
| **PEspatial G** | 1.9B | **DETA**（[[entities/detr]] 系統）| **66.0** |

**シンプルな DETR-style decoder で検出 SOTA を達成した初の汎用対比モデル**。

---

## DINOv3 との関係（再評価）

[[sources/dinov3]] は「PE は dense feature で弱い」と主張していた。しかし PE 論文の PEspatial G は dense 予測で SOTA を取り戻す。両者の対立軸は以下：

| 軸 | PE（特に PEspatial） | DINOv3 |
|---|---|---|
| 学習信号 | テキスト + 画像（事前学習）+ SAM 2.1（空間蒸留） | 画像のみ（[[concepts/self-supervised-learning]]） |
| Dense 用の改良 | **alignment tuning** で末端に特徴を引き出す | **Gram anchoring** で過去の自分にアラインする |
| 空間蒸留の教師 | **SAM 2.1**（教師あり、マスク注釈で訓練済み） | **自分自身の早期スナップショット**（ラベルフリー） |
| データ規模 | 5.4B image-text pairs + 22M videos | 1.689B images |
| 主張 | 「対比学習だけでスケールすれば全タスク解ける」 | 「supervised teacher なしで純粋 SSL でも勝てる」 |

両者は **同じ Meta 内部の競合プロジェクト** として、**WSL（[[concepts/weakly-supervised-pretraining]]）vs SSL** の根源的問いへの異なる答え。**PE のスタンスは「対比学習をシンプルに保ち、SAM のような既存リソースを蒸留教師に使う」**。

### PEspatial の Gram anchoring との対比

[[concepts/gram-anchoring]] と PEspatial の自己蒸留は、目的は近いが手法が異なる：

| | PEspatial の自己蒸留 | DINOv3 Gram Anchoring |
|---|---|---|
| 教師の出所 | **凍結された層 41 特徴**（出力ではない、中間層！） | 過去の自分の出力 Gram 行列 |
| マッチング | 特徴量を直接 cosine マッチング | Gram 行列（特徴量間の関係）のみマッチング |
| 訓練フェーズ | **事後 alignment tuning**（独立段階） | 事前学習中に同時最適化 |
| 外部 supervision | **SAM 2.1（追加教師）必須** | 完全ラベルフリー |

---

## SigLIP / SigLIP 2 との関係

PE と SigLIP は WSL 内で **2 つの対照的戦略** を取る：

| 軸 | CLIP | **SigLIP** | **SigLIP 2** | **PE** |
|---|---|---|---|---|
| 損失 | softmax InfoNCE | sigmoid loss | sigmoid + decoder + 自己蒸留 + マスク予測 | **InfoNCE のまま、頑健化** |
| データ規模 | 4 億（WIT, 2021） | WebLI 大規模 | WebLI 10B / 109 言語 | **5.4B unique pairs / 86B samples seen** |
| バッチサイズ | 32k 必須 | 小〜中で OK | 32k | **131K**（最大） |
| 訓練の特徴 | — | 損失改革 | 全部入りレシピ | **頑健化 + alignment tuning** |
| Dense 予測対応 | 弱 | 弱 | 改善（SILC/TIPS 統合） | **強**（PEspatial、SAM 蒸留 + 自己蒸留） |
| MLLM 対応 | 標準 | 標準 | 標準 | **強**（PElang、専門エンコーダ） |
| 多言語 | 英語のみ | 英語のみ | **109 言語** | 英語のみ |

**戦略の違い**:
- **SigLIP** = 損失関数改革（sigmoid で大バッチ不要、メモリ効率）
- **SigLIP 2** = 既存改善の統合（decoder + 自己蒸留 + マスク予測 + 多言語 + de-bias + NaFlex）
- **PE** = **対比損失のピュアさを保ったまま、頑健化 + alignment tuning で多目的化**

PE と SigLIP 2 は対照的: SigLIP 2 が「複数の事前学習目的を混ぜる」のに対し、PE は「対比学習だけで十分、ただし中間層を活用」を主張。

---

## SAM 3 の backbone として採用

[[sources/sam-3]] / [[entities/sam-3]]（Meta Superintelligence Labs, 2025）は **PE を SAM 3 の共有 backbone として採用**。これは PE が大規模 foundation model で本格採用された初の事例：

- **共有 PE**: 検出器とトラッカーで同じ PE インスタンスを使用
- **テキスト ↔ 画像 aligned** な性質が PCS タスク（テキスト名詞句で全インスタンスを得る）に適合
- DETR ベース detector に PE 出力を fusion encoder で融合
- DINOv3 が「PE は dense task で弱い」と主張した一方、SAM 3 は **fusion encoder + DETR decoder + presence head** という追加構造で PE の弱点を補い、最終的に segmentation で SOTA を達成
- PE が **テキスト条件付き open-vocab セグメンテーションの基盤** として再評価される契機に

これにより PE は「分類用 backbone」から「**PCS タスクの基盤**」へと役割が拡大。さらに本論文の PEspatial で **dense 予測 SOTA そのもの** を達成し、SAM 3 内での PE の役割がさらに重要に。

---

## 弱教師あり vs 自己教師あり: PE と DINOv3 の対比（再掲）

DINOv3 と PE は「**同じ Meta が同時期に出した、対照的なアプローチ**」と見ることもできる：

| 軸 | PE | DINOv3 |
|---|---|---|
| 学習信号 | テキスト + 画像 | 画像のみ |
| データキュレーション | MetaCLIP（テキストキャプションでフィルタ） | 視覚的類似度でフィルタ |
| 強み | global 分類・ゼロショット **＋ MLLM ＋ dense（PEspatial で改善）** | dense 予測・instance 検索・3D |
| データ規模 | 5.4B image-text pairs（86B samples seen） | 1.689B images |
| 公開モデル | PEcore / PElang / PEspatial（各 B/L/G） | ViT-S〜7B + ConvNeXt + dino.txt |

両者は補完的で、実際の応用では**併用**されることも多い（例: VLM の vision tower に両方をマージする手法）。

---

## 主要な貢献まとめ

1. **対比学習だけで dense・MLLM タスクまで解ける** ことを実証（vanilla CLIP では不可能だった）
2. **中間層に一般特徴量が育つ** ことを 3 モデルクラス（contrastive / SSL / captioning）で同時実証
3. **Alignment tuning**（[[concepts/alignment-tuning]]）という新しいファインチューニング戦略を確立
4. **動画データエンジン** で 22M 動画に高品質キャプションを付与、画像性能も同時改善
5. **PVD（PE Video Dataset）** を公開（1M 動画 + 120K 人手精錬キャプション）
6. **頑健な事前学習レシピ** の 9 段階累積アブレーション（progressive resolution が特に効く）

## 関連ページ

- [[sources/perception-encoder]] — 原典の要約
- [[translations/perception-encoder]] — 原典の和訳
- [[concepts/alignment-tuning]] — PE が定義した中間層引き出し戦略
- [[sources/dinov3]] / [[entities/dinov3]] — 純粋 SSL 側の対抗馬
- [[sources/sam-3]] / [[entities/sam-3]] — PE を backbone として採用
- [[entities/clip]] / [[sources/clip]] — PE の理論的祖先
- [[entities/siglip]] / [[sources/siglip]] / [[sources/siglip-2]] — 同系統の Google 競合
- [[concepts/promptable-concept-segmentation]] — PE が SAM 3 で支える新タスク
- [[concepts/weakly-supervised-pretraining]] — PE が属するパラダイム
- [[concepts/contrastive-learning]] — PEcore の事前学習目的
- [[concepts/foundation-model]] — PE は CV foundation model の最新世代
- [[concepts/gram-anchoring]] — DINOv3 が使う、PEspatial の自己蒸留と目的が近い手法
- [[sources/detr]] / [[entities/detr]] — PEspatial が採用する DETA decoder の祖。シンプル DETR-style で COCO 66.0 SOTA を達成
- [[concepts/object-detection]] — PE が SOTA を更新した分野の全体俯瞰
