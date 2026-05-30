---
type: overview
updated: 2026-05-28
---

> 最新更新（2026-05-29）: [[sources/detr]] / [[entities/detr]] / [[concepts/object-detection]] を新規追加（DETR ingest）。さらに [[sources/dino-detector]] / [[entities/dino-detector]] / [[translations/dino-detector]] を追加（DINO 検出器 ingest、SSL の [[entities/dino]] とは別物）。さらに [[sources/glip]] / [[entities/glip]] / [[translations/glip]] を追加（GLIP ingest、open-vocabulary 検出パラダイムの祖）。さらに [[sources/grounding-dino]] / [[entities/grounding-dino]] / [[translations/grounding-dino]] を追加（Grounding DINO ingest、GLIP × DINO 検出器、SAM 3 の直接の祖）。さらに [[sources/yolo-world]] / [[entities/yolo-world]] / [[translations/yolo-world]] を追加（YOLO-World ingest、real-time open-vocab 検出の祖、エッジデバイス向け）。さらに [[sources/grounding-dino-1-5]] / [[entities/grounding-dino-1-5]] / [[translations/grounding-dino-1-5]] を追加（Grounding DINO 1.5 ingest、Pro/Edge 双子スイートで精度と速度を統合）。さらに [[sources/dino-x]] / [[entities/dino-x]] / [[translations/dino-x]] を追加（DINO-X ingest、IDEA Research の unified perception model、Grounding DINO 系統の到達点）。さらに [[sources/internvl]] / [[entities/internvl]] / [[translations/internvl]] を追加（InternVL ingest、6B 視覚 + 8B 言語ミドルウェアで LLM と整列させた初の本格的視覚言語基盤モデル、CVPR 2024）。さらに [[sources/internvl-1-5]] / [[entities/internvl-1-5]] / [[translations/internvl-1-5]] を追加（InternVL 1.5 ingest、26B オープンソース MLLM で GPT-4V との差を 18 ベンチ中 8 SoTA で初めて埋めた、2024 April）。さらに [[sources/mini-internvl]] / [[entities/mini-internvl]] / [[entities/internvit-300m]] / [[translations/mini-internvl]] を追加（Mini-InternVL ingest、InternVL シリーズの「軽量化 + ドメイン特化」分枝、5% パラメータで 90% 性能 + 統一ドメイン適応フレームワーク、2024 Oct）。さらに [[sources/internvl-2-5]] / [[entities/internvl-2-5]] / [[translations/internvl-2-5]] を追加（InternVL 2.5 ingest、MMMU 70% を超えた初のオープン MLLM、1B-78B の 7 サイズ + Progressive Scaling Strategy + Test-Time Scaling、2024 Dec）。さらに [[sources/internvl-3]] / [[entities/internvl-3]] / [[translations/internvl-3]] を追加（InternVL 3 ingest、Native Multimodal Pre-Training への哲学転換 + V2PE + MPO + VisualPRM、MMMU 72.2 で再び SOTA、1B-78B の 7 サイズ、2025 Apr）。さらに [[sources/mpo]] / [[entities/mpo]] / [[entities/mmpr]] / [[translations/mpo]] を追加（MPO + MMPR ingest、MLLM の CoT 推論で性能悪化する現象を初めて解決、DPO + BCO + SFT loss の混合 + 3M 選好データ、InternVL2-8B-MPO で MathVista +8.7、InternVL 3 の Stage 3 で正式採用、2024 Nov）。さらに [[sources/internvl-3-5]] / [[entities/internvl-3-5]] / [[translations/internvl-3-5]] を追加（InternVL 3.5 ingest、Cascade RL = MPO + GSPO の 2 段階強化学習 + MoE スケーリング 241B-A28B + ViR + DvD で 4.05× 加速、MMMU 77.7、GPT-5 との差 3.9%、InternVL シリーズ第 8 世代、2025 Aug）。

# Computer Vision — Overview

このページは Computer Vision（CV, 画像・動画を計算機で理解する研究分野）の全体俯瞰です。原典を ingest するたびに、必要に応じて加筆・修正してください。

> 初学者向けの位置づけ: CV は「画像という高次元・空間構造をもったデータから、意味（クラス・位置・関係・3D 構造・テキスト記述など）を取り出す」分野です。深層学習以前は手作業の特徴量（SIFT, HOG 等）が主流でしたが、2012 年の AlexNet 以降は CNN（畳み込みニューラルネット）と Transformer 系のアーキテクチャが主役になっています。2023 年以降は **CV における基盤モデル**（[[concepts/foundation-model]]）の競争が中核トピックになっています。

## 主要タスク（カテゴリ）

ingest が進んだら、各タスクごとに該当する sources / concepts ページを列挙していきます。

- **画像分類（Image Classification）**：画像 1 枚にラベルを 1 つ付ける。[[entities/imagenet]] が de facto ベンチマーク。
- **物体検出（Object Detection, [[concepts/object-detection]]）**：画像中の物体の位置（バウンディングボックス）とクラスを当てる。古典的には R-CNN ファミリー（Two-stage）/ YOLO・RetinaNet（Single-stage）/ FCOS・CenterNet（Anchor-free）の 3 系統が主流だったが、**2020 年に [[sources/detr]] / [[entities/detr]]（DETR, Carion et al., ECCV 2020）が「集合予測（set prediction）パラダイム」を導入**し NMS / anchor / proposal を排除。**2022 年に [[sources/glip]] / [[entities/glip]]（GLIP, Li et al., CVPR 2022）が「検出 = phrase grounding」として open-vocabulary 検出パラダイムを確立**。**2023 年に [[sources/grounding-dino]] / [[entities/grounding-dino]]（Grounding DINO, Liu et al., ECCV 2024）が GLIP × DINO 検出器の統合で COCO ZS 52.5 AP / ODinW ZS 26.1 SOTA を達成**。**2024 年に [[sources/yolo-world]] / [[entities/yolo-world]]（YOLO-World, Cheng et al., CVPR 2024）が YOLOv8 を open-vocab 化、52 FPS V100 で LVIS 35.4 AP のリアルタイム open-vocab を実現**、同年 5 月に **[[sources/grounding-dino-1-5]] / [[entities/grounding-dino-1-5]]（Grounding DINO 1.5, IDEA Research）が Pro/Edge 双子スイートで COCO ZS 54.3 AP / LVIS-mv 55.7 AP SOTA + Orin NX 10.7 FPS の Edge モデルを同時提供**、同年 11 月に **[[sources/dino-x]] / [[entities/dino-x]]（DINO-X, IDEA Research）が 3 プロンプト + 4 ヘッド + Grounding-100M の unified perception model で LVIS-mv 59.8 / rare APr 63.3 SOTA + Orin NX 20.1 FPS の Edge モデルを実現、「Grounding DINO 系統の到達点」を確立**。**DINOv3 が COCO 検出で凍結バックボーン + Plain-DETR で SOTA mAP 66.1 を達成**（[[sources/dinov3]] §6.3.1）。**PE PEspatial が DETA decoder + Objects365 で COCO 66.0 box AP**（[[sources/perception-encoder]]）でほぼ並ぶ。「DETR ファミリー」（Deformable DETR, DINO-detector, DETA, CoDETR）と「GLIP / Grounding DINO ファミリー（精度志向、unified perception）」（OWL-ViT, SAM 3, GD 1.5 Pro, DINO-X Pro）、「**YOLO-World ファミリー / GD 1.5 Edge / DINO-X Edge（実用志向）**」が現代の三大主流
- **セマンティック / インスタンスセグメンテーション**：画素ごとにクラスを割り当てる。DINOv3 が ADE20K mIoU 63.0 で ONE-PEACE と並ぶ SOTA
- **プロンプト可能セグメンテーション（Promptable Segmentation）**：点・ボックス・テキスト等のプロンプトから任意マスクを返す新タスク。[[entities/sam]] / [[sources/segment-anything]] が CV foundation model の第 3 の系統として確立、[[entities/sam-2]] / [[sources/sam-2]] が動画拡張（PVS: Promptable Visual Segmentation）。詳細: [[concepts/promptable-segmentation]]
- **プロンプト可能コンセプトセグメンテーション（PCS: Promptable Concept Segmentation）**：名詞句または画像 exemplar から、画像/動画中のコンセプトの **全インスタンス** を検出・セグメント・追跡。[[entities/sam-3]] / [[sources/sam-3]] が 2025 年に導入した新タスク。PVS と互補的。詳細: [[concepts/promptable-concept-segmentation]]
- **単眼深度推定（Monocular Depth Estimation）**: 1 枚の画像から各画素の深度を推定。DINOv3 + Depth Anything V2 で複数データセット SOTA
- **3D 理解（Camera pose / Multi-view / View matching）**: VGGT + DINOv3 で全タスク SOTA
- **姿勢推定（Pose Estimation）**
- **画像生成（Image Generation）**：拡散モデル等
- **画像-言語マルチモーダル（VLM, Vision-Language Models）**：CLIP, BLIP, LLaVA など。vision tower に CLIP / SigLIP 2 / PE / DINOv2 / DINOv3 のどれを使うかが議論中
- **3D 復元・NeRF / 3D Gaussian Splatting**：DINOv3 特徴量はパッチ対応の良さで活用される
- **動画理解（Video Understanding）**：DINO/DINOv2/DINOv3 特徴量で訓練なしの動画オブジェクト追跡（DAVIS, YouTube-VOS, MOSE）も対応可能。V-JEPA 2 は動画専用 SSL の SOTA。**SAM 2** ([[entities/sam-2]] / [[sources/sam-2]]) は動画セグメンテーション foundation model として SA-V val/test で先行手法を 14 ポイント差で上回る。**SAM 3** ([[entities/sam-3]] / [[sources/sam-3]]) は名詞句で動画中のコンセプトの全インスタンスを追跡（PCS）
- **インスタンス検索（Instance Recognition）**: Oxford/Paris, Met, AmsterTime 等。DINOv3 が圧倒的に強い領域（Oxford-H で mAP 60.7）
- **自己教師あり学習（Self-Supervised Learning）**：[[concepts/self-supervised-learning]] を参照
- **リモートセンシング・地球観測**：DINOv3 衛星版（[[entities/sat-493m]] で学習）が Geo-Bench 多数で SOTA

## 主要アーキテクチャの系譜

### CNN 系
- LeNet → AlexNet（2012, 深層学習革命の起点）→ VGG → GoogLeNet/Inception → **ResNet（2015）** → DenseNet → EfficientNet → ConvNeXt(2022) → **DINOv3 ConvNeXt 蒸留版（2025）**
- CV における長年の主役。局所性・平行移動不変性という強い帰納バイアスを持つ。

### Transformer 系
- **DETR（2020 May, [[entities/detr]]）** ＝ **CNN backbone + Transformer encoder-decoder で物体検出を end-to-end 化**（CV における Transformer の最初の本格的成功）
- **ViT（2020 Oct, [[concepts/vision-transformer]]）** → DeiT（2021）→ Swin Transformer（2021）→ MAE（2022）→ **iBOT（2022, [[entities/ibot]]）** → **DINOv2（2023, [[entities/dinov2]]）** → **DINOv3（2025, [[entities/dinov3]]）**
- 帰納バイアスが弱い分、データと事前学習が肝。
- 位置埋め込みは learnable → RoPE（[[concepts/rotary-position-embeddings]]）へ移行が進行中（DINOv3, RoPE-ViT, EVA-02 等）
- **DETR ファミリー**: [[entities/detr]]（2020）→ Deformable DETR（2020 Oct）→ DAB-DETR / DN-DETR（2022）→ **[[entities/dino-detector]]（ICLR 2023、COCO 63.3 AP で初の end-to-end Transformer SOTA）** → DETA / CoDETR / Grounding DINO、現代の検出ヘッド標準
- **GLIP / Grounding DINO ファミリー（open-vocabulary 検出、精度志向）**: ViLD（2021）→ MDETR（2021）→ **[[entities/glip]]（CVPR 2022、検出 = phrase grounding 統一）** → OWL-ViT → **[[entities/grounding-dino]]（ECCV 2024、GLIP × DINO 検出器、tight 3-phase fusion）** → **[[entities/grounding-dino-1-5]]（2024 May、ViT-L + Grounding-20M で COCO ZS 54.3 / LVIS-mv 55.7 SOTA、Pro/Edge スイート）** → **[[entities/dino-x]]（2024 Nov、3 プロンプト + 4 ヘッド + Grounding-100M で LVIS-mv 59.8 / rare APr 63.3 SOTA、unified perception model）** → [[entities/sam-3]]（2025、PCS、SAM 3 と並行進化する unified perception）。**「テキストプロンプトで任意のクラスを検出」** という新パラダイム
- **YOLO-World 系 / Grounding DINO 1.5/1.6/DINO-X Edge（real-time open-vocabulary 検出、実用志向）**: **[[entities/yolo-world]]（CVPR 2024、YOLOv8 + CLIP + RepVL-PAN）** が登場。**52 FPS V100 で LVIS 35.4 AP**（GLIP-T の 433× / Grounding-DINO-T の 35× 速度）。Prompt-Then-Detect でテキスト encoder を推論時に削除、テキスト埋め込みをモデル重みに re-parameterize。**エッジデバイスでの open-vocab 検出を初めて現実的に**。同時期に **[[entities/grounding-dino-1-5]] Edge（2024 May、EfficientViT-L1 + Efficient Feature Enhancer）が Orin NX 10.7 FPS で LVIS-mv 36.2 AP**（YOLO-Worldv2-L 32.9 超え）を達成、**Transformer 系のリアルタイム open-vocab 路線**を確立。**[[entities/dino-x]] Edge（2024 Nov、EfficientViT-L2 + Knowledge Distillation + FP16）が Orin NX 20.1 FPS で LVIS-mv 48.3 AP**（YOLO-Worldv2-L を +15.3 AP 凌駕）、実用志向 open-vocab 検出の新 SOTA

### 拡散モデル系
- DDPM（2020）→ Score SDE → Latent Diffusion / **Stable Diffusion（2022）** → SDXL → DiT → Flow Matching
- 画像生成のデファクト。Transformer ベースの DiT に統合される流れ。

### VLM（Vision-Language Models）/ MLLM（Multimodal LLM）
- **CLIP（2021, [[entities/clip]], [[sources/clip]]）** → BLIP → LLaVA → Florence → **InternVL（2023-12, [[entities/internvl]], [[sources/internvl]]、6B 視覚 + 8B QLLaMA 統合）** → **InternVL 1.5（2024-04, [[entities/internvl-1-5]], [[sources/internvl-1-5]]、26B、QLLaMA 廃止 + 動的高解像度 + バイリンガル）** → **Mini-InternVL（2024-10, [[entities/mini-internvl]], [[sources/mini-internvl]]、1B-4B、軽量化 + 統一ドメイン適応）** → **InternVL 2.5（2024-12, [[entities/internvl-2-5]], [[sources/internvl-2-5]]、1B-78B 7 サイズ、MMMU 70% 突破の初 OS MLLM）** → **InternVL 3（2025-04, [[entities/internvl-3]], [[sources/internvl-3]]、Native Multimodal Pre-Training への哲学転換、MMMU 72.2 で再び SOTA）** → **InternVL 3.5（2025-08, [[entities/internvl-3-5]], [[sources/internvl-3-5]]、Cascade RL + MoE + ViR + DvD、MMMU 77.7、GPT-5 との差 3.9%）** → GPT-4V / Gemini-Vision / Claude Sonnet Vision など
- vision tower の選択肢: CLIP, OpenCLIP, **SigLIP 2** ([[entities/siglip]]), **PE** ([[entities/perception-encoder]]), **DINOv2/v3**, **InternViT-6B / V2.5** ([[entities/internvl]] / [[entities/internvl-1-5]] / [[entities/internvl-2-5]] / [[entities/internvl-3]]), **[[entities/internvit-300m\|InternViT-300M / V2.5]]**（軽量版、KD で構築、Mini-InternVL / InternVL 2.5 / InternVL 3 で共有）
- CLIP が CV にゼロショット転移（[[concepts/zero-shot-transfer]]）と対比学習（[[concepts/contrastive-learning]]）を持ち込み、現代マルチモーダル AI の起点となった
- **InternVL** ([[entities/internvl]]) は「視覚エンコーダを 6B、言語ミドルウェアを 8B にスケールアップ」した最初の本格的試みで、ViT-22B（21.7B）を 1/3.7 のパラメータで ADE20K linear probe +12.6 mIoU 上回る。InternVL 1.5 / 2.0 / 2.5 / 3 シリーズの起点
- **InternVL 1.5** ([[entities/internvl-1-5]]) は **「QLLaMA を廃止し LLaVA 系の MLP プロジェクタに統一、動的高解像度（35 種アスペクト比 × 1-40 タイル / 4K）+ バイリンガル（英中）」** という設計で 26B、**18 ベンチ中 8 SoTA、ChartQA / OCRBench / MMBench-CN / CCBench / MathVista で GPT-4V / Gemini Pro / Claude-3 / Qwen-VL-Max を上回る**。「How Far Are We to GPT-4V?」という問いに「OCR と中国語ではもう追いついた」と答えた重要な転換点
- **Mini-InternVL** ([[entities/mini-internvl]]) は **InternVL シリーズの「軽量化 + ドメイン特化」分枝**。[[entities/internvit-300m\|InternViT-300M]]（CLIP-ViT-L 初期化 + InternViT-6B 蒸留）+ 軽量 LLM（Qwen2-0.5B / InternLM2-1.8B / Phi-3-Mini）で **5% パラメータで InternVL2-76B の 90% 性能**。さらに **自律走行 / 医療画像 / リモートセンシング** を VQA 形式に統一する「ドメイン適応フレームワーク」を提案、DA バリアントで GPT-4o / Claude 3.5 Sonnet 等の商用モデルを特化ドメインで凌駕
- **InternVL 2.5** ([[entities/internvl-2-5]]) は **InternVL シリーズの軽量・大規模を統合した最終到達点**。アーキテクチャは [[entities/internvl-1-5\|InternVL 1.5]] と完全に同じ、**訓練戦略・データ品質・テスト時スケーリングのみで MMMU 70% を突破**（70.1、GPT-4o 69.1 / Claude-3.5-Sonnet 68.3 超え）。**1B/2B/4B/8B/26B/38B/78B の 7 サイズ**で消費者 GPU からエンタープライズまでカバー。**Progressive Scaling Strategy 公式化**（小型 LLM で ViT 訓練 → 大型 LLM 転送、Qwen2-VL の 1/12 訓練トークン）。**Test-Time Scaling**（CoT + Majority Voting）で MMMU +3.7。MathVista 72.3 で GPT-4o (63.8) を +8.5 圧倒、RefCOCO 92.3 SOTA、Video-MME / MVBench / MLVU で複数 SOTA。**InternViT 系統で「最終層の線形分離性が低下しつつ open-set 意味を捕捉」を独立発見**（[[entities/perception-encoder\|PE]] と同等の現象、[[concepts/alignment-tuning]] 参照）
- **InternVL 3** ([[entities/internvl-3]]) は **InternVL シリーズの哲学転換**。「LLM Chat 版 → MLLM 事後改造」(InternVL 1.0-2.5) から **「Native Multimodal Pre-Training」**（Qwen2.5 base + テキスト + マルチモーダル共同事前学習）へ。**V2PE**（視覚トークンに位置インクリメント δ < 1）+ **MPO**（DPO + BCO + LM 損失、+4.5 ポイント推論改善）+ **VisualPRM-8B**（Process Reward Model、Best-of-N で小型 +9.9）。**MMMU 72.2 で再び SOTA**（Claude-3.7-Sonnet 75.0 のみ上）、**OCRBench 906 で史上初の 900 超え**、**MathVista 79.0**（GPT-4o 60.0 +19.0）、**VSI-Bench 空間推論で InternVL3-8B が GPT-4o を +8.1 圧倒**。**「マルチモーダル化で言語能力が強くなる」初実証**（Qwen2.5-Chat より +1.6〜+8.9）。**訓練データ完全公開**（`OpenGVLab/InternVL-Data`）、7 サイズ（Qwen2.5 base 系 6 + InternLM3-8B 1）
- 詳細: [[concepts/weakly-supervised-pretraining]] / [[concepts/foundation-model]]

## 自己教師あり学習（SSL）の代表手法系譜

[[concepts/self-supervised-learning]] に詳細あり。

- **対比型**: SimCLR (2020) → MoCo v1/v2/v3 (2019-2021)
- **非対比型（蒸留系）**: BYOL (2020) → SimSiam (2021) → **DINO (2021, [[entities/dino]])** → DINOv2 (2023, [[entities/dinov2]]) → **DINOv3 (2025, [[entities/dinov3]])**
- **クラスタリング型**: DeepCluster → SeLa → SwAV (2020)
- **マスク再構成型 (MIM, [[concepts/masked-image-modeling]])**: BEiT (2021) → **MAE (2021, [[entities/mae]])** → SimMIM (2022)。MIM の理論的祖先は [[concepts/denoising-autoencoder]]（DAE, Vincent 2008）
- **ハイブリッド**: **iBOT (2022, [[entities/ibot]])** = DINO + MIM → DINOv2 → **DINOv3** がこの路線を完成形に
- **JEPA 系**: I-JEPA → V-JEPA → V-JEPA 2 (LeCun 提唱の latent 予測パラダイム)

「**CV における BERT**」探しの本流は、ViT × SSL という組み合わせを軸に進展してきた。

| 年 | マイルストーン |
|---|---|
| 2021 | DINO が「SSL で ViT は emergent properties を持つ」と示す |
| 2022 | iBOT が DINO + MIM のハイブリッドを実現 |
| 2023 | **DINOv2 が iBOT を 1B param × 142M 画像にスケールし、凍結特徴量で OpenCLIP に勝つ** |
| 2025 | **DINOv3 が ViT-7B × LVD-1689M、Gram anchoring で dense feature 劣化を解決、PE/SigLIP 2 に対し dense で広く優位**（ただし PE の PEspatial が後に SAM 2.1 蒸留＋自己蒸留で dense SOTA を一部奪還、最大スケール ADE20k 等では DINOv3 7B が依然優位） |
| 2025 | **Perception Encoder が「対比学習をスケールすると中間層に多目的な一般特徴量が育つ」を発見、alignment tuning で PEcore / PElang / PEspatial の 3 バリアントを構築、検出 SOTA を含む多領域で再リード** |

## 弱教師あり事前学習の系譜

詳細: [[concepts/weakly-supervised-pretraining]] / [[entities/clip]]

- **CLIP（OpenAI, 2021, [[sources/clip]]）**: 4 億画像-テキスト対（[[entities/wit-400m]]）+ 対比学習。ゼロショット分類を可能にし、現代 VLM の礎を作る
- **ALIGN（Google, 2021）** / **OpenCLIP**（LAION）/ **EVA-CLIP** / **MetaCLIP** / **DFN** が続く
- **SigLIP（Google DeepMind, 2023, [[sources/siglip]] / [[entities/siglip]]）**: softmax → sigmoid 損失で **小バッチでも勝利 + メモリ効率改善 + ノイズ頑健**。32k バッチで飽和することを発見、bias term + β₂=0.95 で大バッチ安定化、4 TPU で 1 日訓練可能。SO-400M で 83.2% IN-0
- **SigLIP 2（Google DeepMind, 2025, [[sources/siglip-2]]）**: SigLIP に **LocCa decoder（位置特定）+ SILC/TIPS 自己蒸留＋マスク予測（dense feature）+ ACID 蒸留（小型最適化）+ 多言語化＋ de-bias 化＋ NaFlex（可変アスペクト・解像度）** を統合した「全部入りレシピ」。RefCOCO で +20pt、ADE20k seg で +4.2pt、representation bias を 35.5%→7.3% に削減。g/16 (1B) 新規追加、85.0% IN-0
- **Perception Encoder (PE, Meta, NeurIPS 2025, [[sources/perception-encoder]] / [[entities/perception-encoder]])**: **5.4B unique image-text pairs を 86B samples seen まで訓練** + 22M videos ファインチューン。**「対比学習をスケールするとネットワークの中間層に多目的な一般特徴量が育つ」** という発見と、それを末端に引き出す **alignment tuning**（[[concepts/alignment-tuning]]）で **PEcore（global / ゼロショット SOTA）/ PElang（MLLM 専門）/ PEspatial（SAM 2.1 mask logits + 自己層 41 の 2 教師蒸留 → dense 予測 SOTA）** の 3 バリアントを構築。COCO 検出 66.0 AP_box でシンプル DETR-style decoder で SOTA。JFT-3B / WebLI なしで 3 年ぶりに対比 SOTA 復活
- **InternVL (OpenGVLab Shanghai AI Lab, CVPR 2024, [[sources/internvl]] / [[entities/internvl]])**: **視覚エンコーダを 6B（InternViT-6B）にスケールアップ + 8B 言語ミドルウェア QLLaMA（多言語 LLaMA-7B 初期化）で対比 + 生成 + LLM 接続を 1 モデル統合**。4.98B image-text pairs での 3 段階訓練（contrastive → ITC/ITM/ITG → SFT）で、**ViT-22B（21.7B）を 1/3.7 のパラメータで ADE20K linear probe +12.6 mIoU 上回る**。SigLIP 2 / PE が「対比学習自体を改良」する路線なのに対し、InternVL は「glue layer の 42× スケール（188M → 8B）+ 段階的整列」軸で WSL を発展。InternVL 1.5 / 2.0 / 2.5 / 3 シリーズの起点
- **InternVL 1.5 (OpenGVLab Shanghai AI Lab, 2024-04, [[sources/internvl-1-5]] / [[entities/internvl-1-5]])**: InternVL 1.0 の **MLLM 特化版**。**QLLaMA（8B 言語ミドルウェア）を完全廃止**し LLaVA 系の MLP プロジェクタに統一、InternViT-6B-448px-V1.5（45 層・dynamic 448）+ InternLM2-20B-Chat の 26B 構成。**3 つの改善**: 継続事前学習（InternViT V1.0 → V1.2 → V1.5、後ろから 4 層目が MLLM タスクで最良という発見）+ 動的高解像度（35 種アスペクト比 × 1-40 タイル / 最大 4K）+ Pixel Shuffle で visual token 1/4 圧縮 + バイリンガル（英中）データ + PaddleOCR 擬似 OCR + LLM ベース翻訳パイプライン。**18 ベンチ中 8 SoTA**、**ChartQA / OCRBench / MMBench-CN / CCBench / HallusionBench / MathVista で全商用モデル（GPT-4V / Gemini / Claude-3 / Qwen-VL-Max）を上回る**。「How Far Are We to GPT-4V?」MLLM 競争の風向きを変えた論文
- **Mini-InternVL (OpenGVLab Shanghai AI Lab, 2024-10, [[sources/mini-internvl]] / [[entities/mini-internvl]])**: InternVL シリーズの **「軽量化 + ドメイン特化」分枝**。**[[entities/internvit-300m\|InternViT-300M]]**（CLIP-ViT-L-336px で初期化 + InternViT-6B から **negative cosine similarity** で蒸留）+ 軽量 LLM（**Qwen2-0.5B** / **InternLM2-1.8B** / **Phi-3-Mini**）で **1B/2B/4B クラス** を構築。**5% パラメータで InternVL2-Llama3-76B の 90% 性能**（Mini-InternVL-4B Avg 72.8 vs 76B Avg 81.4）。さらに **自律走行 / 医療画像 / リモートセンシング** を VQA 形式に統一する「ドメイン適応フレームワーク」を提案、Mini-InternVL-DA-4B が **GPT-4o / Claude 3.5 Sonnet を MME-RealWorld 自律走行で +24/+17 圧倒、GMAI-MMBench 医療で LLaVA-Med / Claude3-Opus 超え、DriveLM Challenge で 2B が 26B SOTA に匹敵**。**消費者 GPU での MLLM 展開を現実化** + **「専門 MLLM より、ドメイン適応された汎用 MLLM の方が強い」** という重要な発見
- **InternVL 2.5 (OpenGVLab Shanghai AI Lab, 2024-12, [[sources/internvl-2-5]] / [[entities/internvl-2-5]])**: **MMMU で 70% を超えた初のオープンソース MLLM**（70.1%、GPT-4o 69.1 / Claude-3.5-Sonnet 68.3 / Gemini-1.5-Pro 62.2 を凌駕）。**InternVL シリーズの軽量・大規模を統合した最終到達点**、**1B/2B/4B/8B/26B/38B/78B の 7 サイズスイート** + Pro。アーキテクチャは [[entities/internvl-1-5\|InternVL 1.5]] と完全に同じ ViT-MLP-LLM、視覚は **InternViT-300M-V2.5 / InternViT-6B-V2.5**、LLM は **InternLM 2.5 / Qwen 2.5**。**Progressive Scaling Strategy 公式化**（小型 LLM で ViT 訓練 → 大型 LLM に転送、Qwen2-VL の 1/12 訓練トークン = 120B vs 1.4T）。**Test-Time Scaling**（CoT + Majority Voting）で MMMU +3.7、OS MLLM もテスト時スケーリングの恩恵を受けられることを実証。**MathVista 72.3（GPT-4o +8.5）、RefCOCO 92.3 SOTA、Video-MME 72.1 + MVBench 76.4 + MLVU 75.7（複数 SOTA）、MMB-EN/CN 88.3/88.5、MMStar 69.5 SOTA**。**InternViT 系統で「最終層線形分離性低下 + open-set 意味捕捉」を独立発見**（[[entities/perception-encoder\|PE]] の中間層特徴と同等、2024 年中盤の複数チーム独立観察）
- **InternVL 3 (OpenGVLab Shanghai AI Lab, 2025-04, [[sources/internvl-3]] / [[entities/internvl-3]])**: **InternVL シリーズの哲学転換 + MMMU 72.2 で再び SOTA**。InternVL 1.0-2.5 の **「LLM Chat 版から MLLM 事後改造」** という伝統を捨て、**「Native Multimodal Pre-Training」**（Qwen2.5 base + テキスト + マルチモーダル共同事前学習、200B トークン、1:3 比率）へ転換。**3 つの新技術**: (1) **V2PE**（Variable Visual Position Encoding、視覚トークンに位置インクリメント δ < 1、$\delta=1/4$ で最良）、(2) **MPO**（Mixed Preference Optimization = DPO + BCO + LM 損失、300K 選好データで CoT 推論強化、**+4.5 ポイント改善**）、(3) **VisualPRM-8B**（Visual Process Reward Model、Best-of-N で **小型モデル +9.9 ポイント**）。**1B/2B/8B/9B/14B/38B/78B の 7 サイズ**（Qwen2.5 base 6 + InternLM3-8B 1）。**主要結果**: MMMU **72.2 で SOTA**（Claude-3.7-Sonnet 75.0 のみ上）、**MathVista 79.0**（GPT-4o 60.0 +19.0）、**OCRBench 906**（史上初の 900 超え）、**MME 2549.8 / MMB 89.0/88.7 / MMStar 72.5 / Video-MME 72.7/75.7 / MVBench 78.7 / MLVU 79.5**、**VSI-Bench 空間推論で InternVL3-8B (42.1) が GPT-4o (34.0) を +8.1 圧倒**。**「マルチモーダル化で言語能力が強くなる」の初実証**（Qwen2.5-Chat より +1.6〜+8.9、小型ほど顕著）。**訓練データ + モデル重み完全公開**（`OpenGVLab/InternVL-Data`、open-science 強化）。**弱点**: Visual Grounding で 92.3 → 91.4 退行（grounding データ比率低下）、Claude-3.7-Sonnet MMMU 75.0 / Gemini-2.0-Pro MathVerse 67.3 にはまだ届かない。**OpenAI o1 / DeepSeek R1 系の test-time scaling を MLLM へ本格適用した初の事例**
- **MPO (Wang et al., OpenGVLab Shanghai AI Lab, 2024-11, [[sources/mpo]] / [[entities/mpo]] / [[entities/mmpr]])**: **「MLLM の CoT 推論で性能悪化する」分布シフト問題を初めて解決した論文**。Mixed Preference Optimization = **DPO（相対選好、$w_p=0.8$）+ BCO（絶対品質、$w_q=0.2$）+ SFT loss（生成過程、$w_g=1.0$）** の 3 損失混合。**[[entities/mmpr\|MMPR データセット]]**（3M 選好ペア、6 ドメイン × 18 データ）+ **DropoutNTP**（画像なしで応答補完 → rejected、RLAIF-V の 57.5% コスト）。**10 PO アルゴリズム比較で「DPO + BCO + SFT loss」が最良 CoT 性能** を発見。**InternVL2-8B-MPO で MathVista +8.7（67.0、10× 大きい InternVL2-76B 67.2 と同等）、M3CoT +19.9、MathVision 25.7（当時オープン SOTA）**。**CoT が direct より良くなる** という「正しい挙動」を初実現。**テキスト専用ベンチでも改善**（TheoremQA +5.2、IFEval +4.1） → [[entities/internvl-3\|InternVL 3]] の「Native Pre-Training で言語が強くなる」発見の先例。**InternVL 3 の Stage 3 で正式採用**（+4.1～+4.5 ポイント推論改善）、**InternVL 3.5 では Cascade RL の Stage 1 として進化**（MPO + GSPO の 2 段階）。InternVL シリーズの永続的技術となった
- **InternVL 3.5 (InternVL Team Shanghai AI Lab, 2025-08, [[sources/internvl-3-5]] / [[entities/internvl-3-5]])**: **InternVL シリーズ第 8 世代、商用 GPT-5 との差をオープンソース最小（3.9%）に縮める**。**3 つの中核イノベーション**: (1) **Cascade Reinforcement Learning** = **offline RL (MPO で warm-up) + online RL (GSPO で精緻化)** の 2 段階 coarse-to-fine 戦略、reference model 制約なし、トークン単位の geometric mean importance ratio。**GSPO 単独の半分の GPU 時間で +2.1 ポイント上回る**、全モデルサイズで SFT 単独から **+6-12 ポイント改善**（2B モデル +12.2 最大）、(2) **Visual Resolution Router (ViR) + Visual Consistency Learning (ViCO)**: patch ごとに 1/4 or 1/16 圧縮率を **semantic richness** で動的選択、視覚トークン **50% 削減で性能 99% 維持**、(3) **Decoupled Vision-Language Deployment (DvD)**: ViT/MLP/ViR を vision server、LLM を language server に分離、非同期 3 段階パイプラインで **896 解像度時 4.05× 推論加速**。**MoE スケーリング初導入**: 20B-A4B（GPT-OSS-20B、OpenAI 公開モデル統合）/ 30B-A3B / **241B-A28B（最大、Qwen3-235B-A22B）**。**LLM が Qwen3 base に統一**（InternVL 3 の Qwen2.5 から更新）。**9 サイズ × dense + MoE × Flash 効率版 = 18+ モデル**。**主要結果**: MMMU **77.7（オープンソース新 SOTA）**、MathVista **82.7（GPT-5 81.9 +0.8）**、**VSI-Bench 69.5（GPT-5 37.5 を +32 圧倒、空間推論の新水準）**、WindowsAgentArena でも GPT-4o 圧倒、WebArena-Lite-v2 で GPT-4o の 6× スコア。**Qwen3-base 比で言語 16 ベンチ中 15 で上回る**（[[entities/internvl-3\|InternVL 3]] の「マルチモーダル化で言語が強くなる」発見継続）。**訓練データ + モデル重み完全公開**、Apache 2.0 + Qwen3 ライセンス。**弱点**: Reasoning Overall で GPT-5 に -7.2、Text Overall -6.0、241B 全 load にエンタープライズ級 GPU 必要
- DINOv3 vs PE/SigLIP 2/InternVL は **「テキストなし純粋画像 SSL」vs「テキスト誘導 WSL」** の代表的対立軸。**ただし PE の PEspatial が SAM 2.1 蒸留で dense SOTA を取り戻したため、「dense は SSL の独擅場」という DINOv3 の主張は反論を受けている**（ただし PEspatial は SAM 2.1 という supervised teacher への依存が残る）。InternVL は「視覚スケール側」での反撃（5.9B + 公開 4.98B データで JFT-3B + 21.7B の ViT-22B を超える）

## 基盤モデル（Foundation Model）

詳細: [[concepts/foundation-model]]

- 大規模事前学習 + 汎用下流転用というパラダイム
- NLP: BERT, GPT, PaLM, GPT-4, Claude, Gemini
- CV foundation model の **3 大系統**:
  - **画像-テキスト WSL**: [[entities/clip]] / [[entities/siglip]] / [[entities/perception-encoder]] / [[entities/internvl]] / [[entities/internvl-1-5]] / [[entities/mini-internvl]] / [[entities/internvl-2-5]] / [[entities/internvl-3]] / [[entities/internvl-3-5]]（[[concepts/weakly-supervised-pretraining]]）
  - **純粋 SSL**: [[entities/dinov2]] / [[entities/dinov3]] / [[entities/mae]] / [[entities/ibot]]（[[concepts/self-supervised-learning]]）
  - **モデル支援アノテーション + 教師あり**: [[entities/sam]] / [[entities/sam-2]] / [[entities/sam-3]]（[[concepts/promptable-segmentation]] + [[concepts/promptable-concept-segmentation]]、PVS と PCS の両方）
- その他: Stable Diffusion（生成系）
- **VLM/VLLM/MLLM 統合系**: [[entities/internvl]] が「視覚 + 言語ミドルウェア + LLM を 1 モデル統合する設計」を初めて本格化（CVPR 2024）、[[entities/internvl-1-5]] が **MLLM 特化** に転換し GPT-4V との差を縮める（2024-04、18 ベンチ中 8 SoTA）、[[entities/mini-internvl]] が **軽量 + ドメイン特化** に分岐（2024-10、1B-4B、5%/90%、ドメイン適応フレームワーク）、**[[entities/internvl-2-5]] が軽量・大規模を統合**（2024-12、1B-78B 7 サイズ、**MMMU 70% 突破の初 OS MLLM**、Progressive Scaling Strategy + Test-Time Scaling 公式化）、**[[entities/internvl-3]] が哲学転換**（2025-04、**Native Multimodal Pre-Training + V2PE + MPO + VisualPRM**、MMMU 72.2 で再び SOTA、Qwen2.5-Chat より言語が強い）、**[[entities/internvl-3-5]] が Cascade RL + MoE スケーリング**（2025-08、9 サイズ × dense + MoE × Flash、**MMMU 77.7、GPT-5 との差 3.9%**、ViR + DvD で 4.05× 加速、VSI-Bench で GPT-5 を +32 圧倒）。後の Qwen-VL 系の路線

## 主要技術トピック

- **Set Prediction Paradigm** ([[concepts/object-detection]] / [[sources/detr]]): DETR が導入した「物体検出を集合予測問題として扱う」パラダイム。Hungarian 二部マッチングで NMS / anchor / proposal を排除、現代の検出ヘッドの標準
- **Gram Anchoring** ([[concepts/gram-anchoring]]): DINOv3 で導入、長時間学習での dense feature 劣化を防ぐ新しい SSL 正則化
- **Alignment Tuning** ([[concepts/alignment-tuning]]): Perception Encoder で導入。事前学習済みエンコーダの中間層に眠る一般特徴量を、短いファインチューニング段階で最終層に引き出す戦略。**「対比学習で訓練された CLIP モデルの最良の特徴は最終層ではなく中間層にある」** という発見がきっかけ。PE では言語アラインメント（PElang, Llama decoder 統合）と空間アラインメント（PEspatial, SAM 2.1 mask logits + 自己層 41 の 2 教師蒸留）の 2 種で具体化
- **RoPE** ([[concepts/rotary-position-embeddings]]): NLP 由来の回転位置埋め込み、可変解像度対応に必須
- **Multi-student distillation**: DINOv3 で実装、1 teacher の forward を複数 student で共有する効率化
- **Register tokens**: Darcet et al. 2023 → DINOv2 with registers / DINOv3 で標準採用
- **Progressive resolution**: 訓練中に解像度を 98 → 154 → 224 → 336 → 448 と段階的に上げる戦略。Perception Encoder の 9 段階アブレーションで COCO 検出（凍結特徴）+10 mAP という最大の効果。グローバル token への過適合を防ぐ役割と推測

## 主要データセット・ベンチマーク

- **画像分類**: ImageNet ([[entities/imagenet]]), CIFAR-10/100, iNaturalist, Places205, ObjectNet
- **検出**: COCO, COCO-O, Pascal VOC, Open Images, Objects365
- **セグメンテーション**: ADE20K, Cityscapes, PASCAL VOC12, COCO-Stuff
- **画像検索**: Oxford / Paris, GLDv2, Met, AmsterTime
- **動画**: DAVIS, YouTube-VOS, MOSE, Kinetics-400, UCF-101, Something-Something v2
- **深度推定**: NYUv2, KITTI, ETH3D, ScanNet, SUN RGB-D, DIODE
- **ロバストネス**: ImageNet-V2 / -A / -R / -C / -Sketch
- **公平性**: Dollar Street, Casual Conversations
- **地球観測**: Geo-Bench, LoveDA, iSAID, DIOR, SatLidar, Open-Canopy
- **大規模事前学習**: JFT-300M (Google内部), LAION-2B/5B, **LVD-142M ([[entities/lvd-142m]], Meta内部)**, **LVD-1689M ([[entities/lvd-1689m]], Meta内部)**, **SAT-493M ([[entities/sat-493m]], Meta内部)**, DataComp, **WIT-400M ([[entities/wit-400m]], OpenAI内部、CLIP の訓練データ)**, WebLI (Google内部), **SA-1B ([[entities/sa-1b]], Meta公開、SAM の訓練データ、11M 画像 × 1.1B マスク)**, **SA-V ([[entities/sa-v]], Meta公開 CC by 4.0、SAM 2 の訓練データ、50.9K 動画 × 642.6K masklet × 35.5M マスク)**, **SA-Co ([[entities/sa-co]], Meta公開、SAM 3 の訓練・評価データ、5.2M 画像 + 4M unique NP + 52M マスク + benchmark 207K concepts)**

## 未整理メモ

（今後の調査テーマ、open question、気になっている論文タイトル等）

- MoCo / SwAV / BEiT 単独ページ化（系譜の理解を深めるため）
- SAM 1/2/3 はいずれも ingest 済み（[[sources/segment-anything]] / [[sources/sam-2]] / [[sources/sam-3]]）
- BYOL / MAE / PE / DETR / DINO-detector / GLIP / Grounding DINO / YOLO-World / Grounding DINO 1.5 / DINO-X / InternVL / InternVL 1.5 / Mini-InternVL / InternVL 2.5 / InternVL 3 / MPO / InternVL 3.5 はいずれも ingest 済み（[[sources/byol]] / [[sources/mae]] / [[sources/perception-encoder]] / [[sources/detr]] / [[sources/dino-detector]] / [[sources/glip]] / [[sources/grounding-dino]] / [[sources/yolo-world]] / [[sources/grounding-dino-1-5]] / [[sources/dino-x]] / [[sources/internvl]] / [[sources/internvl-1-5]] / [[sources/mini-internvl]] / [[sources/internvl-2-5]] / [[sources/internvl-3]] / [[sources/mpo]] / [[sources/internvl-3-5]]）
- AIMv2（Apple のキャプション化型事前学習、PE の主要比較相手）の ingest 候補
- InternVideo2（動画ネイティブ事前学習、PE の主要動画ベースライン）の ingest 候補
- AM-RADIO（複数 foundation model 蒸留統合、PE alignment tuning と思想が近い）の ingest 候補
- Deformable DETR / DETA / CoDETR / MDETR / OWL-ViT / ViLD / DetCLIPv2 / DetCLIPv3 / GLIPv2 / MM-Grounding-DINO / YOLO-Worldv2 / T-Rex2 / APE / GLEE-Pro / OmDet-Turbo / Grounded SAM / Mask2Former / Mask DINO / ED-Pose / Osprey — DETR / open-vocab 検出 / unified perception ファミリーの後続論文群の ingest 候補
- DINOv2 with registers / register tokens 単独論文の ingest 候補
- Probe3D, VGGT 等の 3D 系論文の ingest 候補
- 拡散モデル系（DDPM, LDM）の ingest 候補
- VLM 系（LLaVA, BLIP-2）の ingest 候補
- V-JEPA / V-JEPA 2 の ingest 候補（動画 SSL）

---

関連: [[index]]
