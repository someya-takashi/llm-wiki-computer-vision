---
type: overview
updated: 2026-05-28
---

> 最新更新（2026-05-28）: [[sources/detr]] / [[entities/detr]] / [[concepts/object-detection]] を新規追加（DETR ingest）。さらに [[sources/dino-detector]] / [[entities/dino-detector]] / [[translations/dino-detector]] を追加（DINO 検出器 ingest、SSL の [[entities/dino]] とは別物）。さらに [[sources/glip]] / [[entities/glip]] / [[translations/glip]] を追加（GLIP ingest、open-vocabulary 検出パラダイムの祖）。さらに [[sources/grounding-dino]] / [[entities/grounding-dino]] / [[translations/grounding-dino]] を追加（Grounding DINO ingest、GLIP × DINO 検出器、SAM 3 の直接の祖）。さらに [[sources/yolo-world]] / [[entities/yolo-world]] / [[translations/yolo-world]] を追加（YOLO-World ingest、real-time open-vocab 検出の祖、エッジデバイス向け）。さらに [[sources/grounding-dino-1-5]] / [[entities/grounding-dino-1-5]] / [[translations/grounding-dino-1-5]] を追加（Grounding DINO 1.5 ingest、Pro/Edge 双子スイートで精度と速度を統合）。さらに [[sources/dino-x]] / [[entities/dino-x]] / [[translations/dino-x]] を追加（DINO-X ingest、IDEA Research の unified perception model、Grounding DINO 系統の到達点）。

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

### VLM（Vision-Language Models）
- **CLIP（2021, [[entities/clip]], [[sources/clip]]）** → BLIP → LLaVA → Florence → GPT-4V / Gemini-Vision / Claude Sonnet Vision など
- vision tower の選択肢: CLIP, OpenCLIP, **SigLIP 2** ([[entities/siglip]]), **PE** ([[entities/perception-encoder]]), **DINOv2/v3**
- CLIP が CV にゼロショット転移（[[concepts/zero-shot-transfer]]）と対比学習（[[concepts/contrastive-learning]]）を持ち込み、現代マルチモーダル AI の起点となった
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
- DINOv3 vs PE/SigLIP 2 は **「テキストなし純粋画像 SSL」vs「テキスト誘導 WSL」** の代表的対立軸。**ただし PE の PEspatial が SAM 2.1 蒸留で dense SOTA を取り戻したため、「dense は SSL の独擅場」という DINOv3 の主張は反論を受けている**（ただし PEspatial は SAM 2.1 という supervised teacher への依存が残る）

## 基盤モデル（Foundation Model）

詳細: [[concepts/foundation-model]]

- 大規模事前学習 + 汎用下流転用というパラダイム
- NLP: BERT, GPT, PaLM, GPT-4, Claude, Gemini
- CV foundation model の **3 大系統**:
  - **画像-テキスト WSL**: [[entities/clip]] / [[entities/siglip]] / [[entities/perception-encoder]]（[[concepts/weakly-supervised-pretraining]]）
  - **純粋 SSL**: [[entities/dinov2]] / [[entities/dinov3]] / [[entities/mae]] / [[entities/ibot]]（[[concepts/self-supervised-learning]]）
  - **モデル支援アノテーション + 教師あり**: [[entities/sam]] / [[entities/sam-2]] / [[entities/sam-3]]（[[concepts/promptable-segmentation]] + [[concepts/promptable-concept-segmentation]]、PVS と PCS の両方）
- その他: Stable Diffusion（生成系）

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
- BYOL / MAE / PE / DETR / DINO-detector / GLIP / Grounding DINO / YOLO-World / Grounding DINO 1.5 / DINO-X はいずれも ingest 済み（[[sources/byol]] / [[sources/mae]] / [[sources/perception-encoder]] / [[sources/detr]] / [[sources/dino-detector]] / [[sources/glip]] / [[sources/grounding-dino]] / [[sources/yolo-world]] / [[sources/grounding-dino-1-5]] / [[sources/dino-x]]）
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
