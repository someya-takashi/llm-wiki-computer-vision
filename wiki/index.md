# Index — Computer Vision Wiki

このページは wiki 全体のカタログです。ingest / query で新しいページを作成するたびに、対応するセクションに 1 行追記してください。

書式: `- [[<slug>]] — <一行の説明>`

---

## Overview

- [[overview]] — Computer Vision 分野全体の俯瞰（随時更新）

## Sources

### Papers

- [[sources/dino-emerging-properties-in-self-supervised-vit]] — DINO（Caron et al., 2021/ICCV）。ラベルなしの自己蒸留として ViT を学習。教師なしで自己注意マップにオブジェクト境界が現れることを発見。
- [[sources/dinov2-learning-robust-visual-features-without-supervision]] — DINOv2（Oquab et al., 2023/TMLR 2024）。iBOT を 1B パラメータ × 142M キュレーション画像に scale。凍結特徴量のまま OpenCLIP に勝つ純粋 SSL 基盤モデル。
- [[sources/dinov3]] — DINOv3（Siméoni et al., 2025）。ViT-7B × LVD-1689M に scale、Gram anchoring で dense feature 劣化を解決。凍結バックボーンで PE / SigLIP 2 に対し dense タスクで広く優位、SOTA 多数更新。**ただし PE PEspatial が後に SAM 2.1 蒸留 + 自己蒸留で dense SOTA を部分的に奪還、最大スケール ADE20k 等では DINOv3 7B が依然優位**（[[sources/perception-encoder]] 参照）。
- [[sources/mae]] — MAE（He et al., 2021/CVPR 2022）。「画像の 75% をマスクして残り 25% から画素を再構成」というシンプルな SSL。非対称 encoder-decoder で ViT-H を IN1K のみで 87.8% 達成。CV MIM の中核手法。
- [[sources/ibot]] — iBOT（Zhou et al., 2021/ICLR 2022）。**online tokenizer** を提案し、DINO の自己蒸留と MIM を統合。DINOv2/DINOv3 の損失関数の直接の源。
- [[sources/clip]] — CLIP（Radford et al., 2021/ICML 2021）。4 億画像-テキスト対 + 対比学習で「a photo of a {class}」プロンプトによるゼロショット分類を実現。現代マルチモーダル AI の起点。
- [[sources/segment-anything]] — SAM（Kirillov et al., 2023/ICCV 2023）。promptable segmentation という新タスクを定義し、データエンジンで 1.1B マスクを構築。CV における初の本格的セグメンテーション基盤モデル。
- [[sources/sam-2]] — SAM 2（Ravi et al., 2024/ICLR 2025）。SAM を動画に拡張、streaming memory + Hiera 画像エンコーダ。SA-V（35.5M マスク / 642.6K masklet / 50.9K 動画、CC by 4.0）を構築。画像でも SAM v1 比 6× 高速。
- [[sources/sam-3]] — SAM 3（Carion et al., 2025）。新タスク PCS（Promptable Concept Segmentation）を導入、名詞句/画像 exemplar で全インスタンスを検出・セグメント・追跡。Perception Encoder backbone + DETR ベース detector + presence head + SAM 2 風 tracker。SA-Co（4M unique NP / 52M マスク、benchmark 207K concepts）を構築。
- [[sources/siglip]] — SigLIP（Zhai et al., 2023/ICCV 2023）。CLIP の softmax 対比損失を sigmoid 損失に置き換え、小バッチで圧倒的に勝つ + メモリ効率改善。4 TPU で 1 日訓練して 79.7% IN-0、SO/400M で 83.2%。バッチサイズが 32k で飽和することを発見。bias term + β₂=0.95 で大バッチ安定化。
- [[sources/siglip-2]] — SigLIP 2（Tschannen et al., 2025）。SigLIP に **LocCa decoder + SILC/TIPS 自己蒸留＋マスク予測 + ACID 蒸留 + 多言語＋de-bias + NaFlex** を統合した「全部入りレシピ」。RefCOCO で +20pt、ADE20k seg で +4.2pt、representation bias を 35.5%→7.3% に削減。g/16 (1B) 新サイズで 85.0% IN-0。WebLI 10B 画像 / 109 言語で訓練。
- [[sources/simclr]] — SimCLR（Chen et al., ICML 2020）。4 コンポーネント（データ拡張 + エンコーダ + 非線形射影ヘッド + NT-Xent 損失）を体系的なアブレーションで解明。ResNet-50 (4×) で 76.5% top-1（教師あり ResNet-50 と同等）、1% ラベルで top-5 85.8%。対比 SSL の 2020 年標準ベースライン。
- [[sources/byol]] — BYOL（Grill et al., NeurIPS 2020）。オンライン×ターゲット 2 ネットワーク + predictor で負例なしに 74.3% top-1 達成（SimCLR+5pt）。「負例は必須か？」に「否」と答えた非対比 SSL の先駆け。
- [[sources/mixmatch]] — MixMatch（Berthelot et al., NeurIPS 2019）。一貫性正則化 + エントロピー最小化 + MixUp を統合した半教師あり学習。CIFAR-10（250 ラベル）で 11.08%（VAT の 3.3 倍改善）。
- [[sources/fixmatch]] — FixMatch（Sohn et al., NeurIPS 2020）。弱→強の非対称拡張 + 信頼度閾値 τ=0.95 だけで MixMatch を大幅更新。CIFAR-10（250 ラベル）で 5.07%。半教師あり学習の設計原則を確立。
- [[sources/flexmatch]] — FlexMatch（Zhang et al., NeurIPS 2021）。CPL（クラス別動的閾値）で FixMatch を拡張。CIFAR-10（40 ラベル）で 4.97%、収束速度 1/5。ただし SVHN（クラス不均衡）では逆に悪化。
- [[sources/revisiting-ssl-foundation-models]] — Revisiting SSL in the Era of Foundation Models（Zhang et al., NeurIPS 2025）。VFM 時代に FixMatch/FlexMatch/SoftMatch が Labeled-only PEFT を凌駕できないことを実証、V-PET（VFM × PEFT アンサンブル疑似ラベリング）を提案。SeSL 研究の前提を再定義。
- [[sources/perception-encoder]] — Perception Encoder（Bolya et al., NeurIPS 2025）。**「対比学習をスケールするとネットワークの中間層に多目的な一般特徴量が育つ」** ことを発見、`alignment tuning`（[[concepts/alignment-tuning]]）で末端に引き出して PEcore（ゼロショット SOTA）/ PElang（MLLM 専門）/ PEspatial（dense 予測 SOTA）の 3 バリアントを構築。5.4B image-text pairs + 22M videos + 86B samples seen。COCO 検出 66.0 AP_box でシンプル DETR-style decoder で SOTA 達成。
- [[sources/detr]] — DETR（Carion et al., ECCV 2020）。**物体検出を「集合予測（set prediction）問題」として再定義**、Hungarian アルゴリズムによる二部マッチング + Transformer encoder-decoder + 並列デコーディングで **NMS / anchor / proposal を排除**。COCO で Faster R-CNN と同等の 42 AP、大物体で +7.8 AP。Deformable DETR / DAB-DETR / DN-DETR / DINO-detector / DETA / CoDETR / MDETR という DETR ファミリーの祖。SAM 3 と PE PEspatial の検出ヘッドの源流。
- [[sources/dino-detector]] — DINO 検出器（Zhang et al., ICLR 2023）。**DETR ファミリーの集大成**。DAB-DETR + DN-DETR + Deformable DETR を統合し、3 つの新技法（**Contrastive DeNoising / Mixed Query Selection / Look Forward Twice**）を追加。ResNet-50 12 epoch で 49.4 AP（DN-DETR +6.0）、**SwinL + Objects365 で COCO test-dev 63.3 AP** — 初の end-to-end Transformer SOTA。**SSL の DINO（[[sources/dino-emerging-properties-in-self-supervised-vit]]）とは完全に別物**。
- [[sources/glip]] — GLIP（Li et al., CVPR 2022）。**物体検出を phrase grounding として再定式化**することで、box 分類ロジットを region-word alignment スコアに置換。**language-aware deep fusion (X-MHA)** + **self-training で 24M の web 画像-テキストペアに自動 box 注釈**（計 27M grounding データ）。COCO ゼロショット **49.8 AP**（Faster RCNN 教師ありを上回る）、fine-tune **61.5 AP**（SOTA）、LVIS rare クラスで教師あり MaskRCNN 超え。**open-vocabulary 検出パラダイムを確立**、Grounding DINO / SAM 3 / OWL-ViT の祖。
- [[sources/grounding-dino]] — Grounding DINO（Liu et al., ECCV 2024）。**「GLIP × DINO 検出器」**: GLIP の grounded pre-training を [[entities/dino-detector|DINO 検出器]] に移植。**Tight 3-Phase Fusion**（Phase A neck + B query init + C decoder すべてで融合、GLIP の A のみ・OV-DETR の B のみを統合）、**Language-Guided Query Selection**、**Sub-Sentence Level Text Representation** が新工夫。COCO zero-shot **52.5 AP**（GLIP-L 49.8 超え）、fine-tune **63.0 AP**、ODinW zero-shot **26.1 mean AP**（Florence 841M を 341M で上回る SOTA）。**REC（指示表現理解）を open-set 評価に拡張**。**SAM 3 と MM-Grounding-DINO の直接の祖**。
- [[sources/yolo-world]] — YOLO-World（Cheng et al., CVPR 2024）。**「YOLO + open-vocabulary 検出 = リアルタイム open-vocab 検出」**: YOLOv8 を CLIP text encoder + 新ネック **RepVL-PAN**（**Text-guided CSPLayer with max-sigmoid attention** + **Image-Pooling Attention 27 patch tokens**）で open-vocab 化。**Prompt-Then-Detect パラダイム**でテキスト encoder を推論時に削除、テキスト埋め込みをモデル重みに **re-parameterize**。**LVIS ゼロショット 35.4 AP at 52 FPS V100**（Grounding DINO の 34.7× / GLIP の 433× 速度）。**Region-Text Contrastive Loss** で検出 + grounding + image-text データを統一。**GLIP-L で CC3M に疑似ラベリング**（246K 画像 / 821K 注釈）。**CLIP text encoder が決定的**（BERT 比 APr +10.1 AP）。「精度志向の GLIP / Grounding DINO」に対する「実用志向」の対抗路線、**エッジデバイスでの open-vocab 検出を可能に**。
- [[sources/grounding-dino-1-5]] — Grounding DINO 1.5（Ren et al., IDEA Research, 2024 May）。**[[sources/grounding-dino|Grounding DINO]] の Pro/Edge 双子拡張**。論文タイトルの "Edge" は **「分野の先端」と「エッジコンピューティング」の二重の意味**。**Pro**: ViT-L backbone + **Grounding-20M** 訓練データ + early fusion + 改善された負例サンプリング。**COCO ZS 54.3 AP / LVIS-minival 55.7 AP**（DetCLIPv3 SOTA +6.9 AP）/ **ODinW35 30.2 AP** SOTA。**Edge**: EfficientViT-L1 + **Efficient Feature Enhancer**（P5 のみ cross-modality 融合 + vanilla self-attn + cross-scale fusion）。**A100 TRT 75.2 FPS / Orin NX 10.7 FPS、LVIS-mv 36.2 AP**（YOLO-Worldv2-L 32.9 超え）。**精度志向と実用志向を 1 モデルスイートで統合**。
- [[sources/dino-x]] — DINO-X（IDEA Research, 2024 Nov）。**[[sources/grounding-dino-1-5|Grounding DINO 1.5]] の正統な後継 = unified object-centric vision model**。**3 種プロンプト**（Text [CLIP encoder]/Visual [T-Rex2 流]/Customized [prompt-tuning]）+ **4 種 perception head**（Box/Mask [Mask2Former 流]/Keypoint [ED-Pose 簡略]/Language [OPT-125M 軽量自己回帰]）+ **Grounding-100M**（GD 1.5 の 5×）。**Pro (ViT-L)**: COCO ZS 56.0 / LVIS-mv 59.8 / **LVIS rare APr 63.3**（GD 1.6 Pro から +5.8、長尾劇的改善）/ Visual Genome region caption CIDEr **201.8 SOTA**。**Edge (EfficientViT-L2)**: Knowledge Distillation (Pro→Edge) + FP16 量子化で **Orin NX 20.1 FPS**（GD 1.5 Edge から +87%）かつ LVIS-mv **48.3 AP**（YOLO-Worldv2-L 33.0 を +15.3）。**Universal Object Prompt で prompt-free 検出**という新タスク。**Grounding DINO 系統の到達点**。

### Articles

（まだありません）

## Translations

- [[translations/dino-emerging-properties-in-self-supervised-vit]] — DINO 原論文（§1-6 + Abstract）の全文和訳。Appendix と References は除外。
- [[translations/dinov2-learning-robust-visual-features-without-supervision]] — DINOv2 原論文（§1-10 + Abstract）の全文和訳。Appendix と References は除外。
- [[translations/dinov3]] — DINOv3 原論文（§1-10 + Abstract）の全文和訳。Appendix と References は除外。
- [[translations/mae]] — MAE 原論文の全文和訳。**Appendix A〜C を含む**（References のみ除外）。
- [[translations/ibot]] — iBOT 原論文の全文和訳。**Appendix A〜G を含む**（References のみ除外）。
- [[translations/clip]] — CLIP 原論文の全文和訳。**Appendix A〜F を含む**（References のみ除外）。
- [[translations/segment-anything]] — SAM 原論文の全文和訳。**Appendix A〜G を含む**（References のみ除外）。
- [[translations/sam-2]] — SAM 2 原論文の全文和訳。**Appendix A〜G を含む**（References のみ除外）。
- [[translations/sam-3]] — SAM 3 原論文の全文和訳。Abstract + §1-9 + Appendix A.1 部分のみ（原典ファイルに残部の Appendix が含まれないため、ユーザー判断で本文のみ ingest）。
- [[translations/siglip]] — SigLIP 原論文の全文和訳。本文 §1-5 + Acknowledgements（独立 Appendix なし、PDF は本文 10 ページ）。
- [[translations/siglip-2]] — SigLIP 2 原論文の全文和訳。**Appendix A-C（PaliGemma 全結果 / NaFlex 全結果 / 文化的多様性・公平性全結果）を含む**（References のみ除外）。
- [[translations/simclr]] — SimCLR 原論文の全文和訳。Abstract + §1-8 + Appendix A-C（拡張詳細 / 追加実験 / 関連手法詳細比較）を含む（References のみ除外）。
- [[translations/byol]] — BYOL 原論文の全文和訳。Abstract + §1-6 + Appendix A-B（アルゴリズム / 画像拡張詳細）を含む（References のみ除外）。
- [[translations/mixmatch]] — MixMatch 原論文の全文和訳。Abstract + §1-5 + Appendix A-C（記号定義 / 全数値表 / 13 層 ConvNet 結果）を含む（References のみ除外）。
- [[translations/fixmatch]] — FixMatch 原論文の全文和訳。Abstract + §1-6 + Broader Impact + Appendix A-E（アルゴリズム擬似コード / 全数値表・ハイパーパラメータ / ImageNet 実装 / 拡張手法 / データ変換詳細）を含む（References のみ除外）。
- [[translations/flexmatch]] — FlexMatch 原論文の全文和訳。Abstract + §1-6 + Broader Impact + Appendix A-B（ハイパーパラメータ / クラスごと精度 / 中央値エラー率 / TorchSSL ベンチマーク全 4 データセット）を含む（References のみ除外）。
- [[translations/revisiting-ssl-foundation-models]] — Revisiting SSL in the Era of Foundation Models 原論文の本文和訳。Abstract + §1-7 + Acknowledgment（References と Appendix A-C は除外、ユーザー指示）。図 1-6 を `<figure>` で埋め込み。
- [[translations/perception-encoder]] — Perception Encoder 原論文の本文和訳。Abstract + §1-7 + Additional Contributors（References と Appendix A-F は除外、ユーザー指示）。図 1-10 を `<figure>` で埋め込み。
- [[translations/detr]] — DETR 原論文の本文和訳。Abstract + §1-6（Acknowledgements 含む、References と Appendix 0.A は除外）。図 1-9 を `<figure>` で埋め込み（図 5 は markdown に画像なし）。
- [[translations/dino-detector]] — DINO 検出器原論文の本文和訳。Abstract + §1-5（References と Appendix 0.A-0.D は除外、ユーザー指示）。図 1-8 を `<figure>` で埋め込み、表 1-4 を含む。
- [[translations/glip]] — GLIP 原論文の本文和訳。Abstract + §1-6 + Acknowledgement（References と Appendix A-E は除外、ユーザー指示）。図 1, 2, 4-7 を `<figure>` で埋め込み（図 3, 8 は markdown に画像なし）。表 1-5 を含む。
- [[translations/grounding-dino]] — Grounding DINO 原論文の本文和訳。Abstract + §1-6（Acknowledgement 含む、References と Appendix A-I は除外）。図 1-5 を `<figure>` で埋め込み、表 1-7 を含む。Algorithm 1（language-guided query selection の PyTorch コード）を含む。
- [[translations/yolo-world]] — YOLO-World 原論文の本文和訳。Abstract + §1-5（References と Appendix A-C は除外、ユーザー指示）。図 1-7 を `<figure>` で埋め込み、表 1-8 を含む。RepVL-PAN（T-CSPLayer + I-Pooling Attention）、Prompt-Then-Detect、Region-Text Contrastive Loss、CLIP vs BERT ablation 等の核心を含む。
- [[translations/grounding-dino-1-5]] — Grounding DINO 1.5 原論文の本文和訳。Abstract + §1-6（Acknowledgement 含む、References と §7 Appendix は除外）。図 1-16 を `<figure>` で埋め込み、表 1-5 を含む。Pro と Edge 両モデルのアーキテクチャ詳細、Grounding-20M、COCO/LVIS/ODinW の全結果、定性的可視化を含む。
- [[translations/dino-x]] — DINO-X 原論文の本文和訳。Abstract + §1-7（Acknowledgement 含む、References は除外）。図 1-11 を `<figure>` で埋め込み、表 1-7 を含む。Pro と Edge 両モデル、4 種ヘッド（Box/Mask/Keypoint/Language）、3 種プロンプト、Grounding-100M、Knowledge Distillation、FP16 量子化、prompt-free 検出、Side-by-side 比較を含む。

## Concepts

- [[concepts/self-supervised-learning]] — SSL（自己教師あり学習）の全体像。対比型・非対比型・クラスタリング型・マスク再構成型・ハイブリッド型、momentum encoder と multi-crop の解説を含む。
- [[concepts/vision-transformer]] — ViT のアーキテクチャ。パッチ分割、[CLS] トークン、命名規約（ViT-S/B/L、/16・/8）、CNN との比較。
- [[concepts/knowledge-distillation]] — 知識蒸留の基礎（dark knowledge, soft target, 温度付き softmax）と DINO 流 self-distillation との対比。
- [[concepts/knn-evaluation-protocol]] — SSL の表現品質を測る k-NN 評価。なぜ DINO で特に注目されたか。
- [[concepts/masked-image-modeling]] — MIM（BERT 流マスク予測の画像版）。BEiT, MAE, iBOT, SimMIM の系譜。
- [[concepts/denoising-autoencoder]] — DAE。MAE / BERT MLM / 拡散モデルなど現代 SSL の理論的祖先。「破損 → 再構成」枠組み。
- [[concepts/foundation-model]] — 基盤モデルの定義と歴史。NLP（BERT/GPT）から CV（CLIP, DINOv2, DINOv3）への系譜、必要要件と批判。
- [[concepts/weakly-supervised-pretraining]] — 弱教師あり事前学習（CLIP, ALIGN, OpenCLIP, SigLIP, PE）。SSL との対比、長所と弱点。
- [[concepts/gram-anchoring]] — DINOv3 が提案した、パッチ間 Gram 行列を過去の teacher に合わせる正則化。dense feature 劣化問題の解決法。
- [[concepts/rotary-position-embeddings]] — RoPE。NLP で標準の位置埋め込みを vision に持ち込み。可変解像度対応に必須。
- [[concepts/online-tokenizer]] — iBOT が提案した、teacher 自身を MIM の視覚トークナイザとして動的に使う発想。DINOv2/v3 の損失設計の中核。
- [[concepts/contrastive-learning]] — 対比学習。正例ペアを近づけ負例を遠ざける。InfoNCE 損失、SimCLR/MoCo/CLIP/DINO 系統の基盤。
- [[concepts/semi-supervised-learning]] — 半教師あり学習（SeSL）。少量ラベルあき + 大量ラベルなしを組み合わせる。一貫性正則化 / エントロピー最小化 / MixUp の 3 系統。自己教師あり学習（SSL）との違いを解説。
- [[concepts/parameter-efficient-fine-tuning]] — PEFT（パラメータ効率的ファインチューニング）。VFM の一部だけを更新する LoRA/AdaptFormer/BitFit/VPT 等の手法群。VFM 時代の SeSL の主役。
- [[concepts/zero-shot-transfer]] — ゼロショット転移。CLIP が CV に持ち込んだ「プロンプトのみで追加訓練なしの推論」というパラダイム。
- [[concepts/promptable-segmentation]] — SAM が定義した「任意プロンプトから妥当マスクを返す」新タスク（PVS）。セグメンテーション基盤モデルの事前学習目的兼ゼロショット転移インターフェイス。
- [[concepts/promptable-concept-segmentation]] — SAM 3 が定義した「名詞句/画像 exemplar からコンセプトの全インスタンスを返す」新タスク（PCS）。PVS と互補的な open-vocabulary な軸。
- [[concepts/alignment-tuning]] — Perception Encoder が定義したファインチューニング戦略。「事前学習済みエンコーダの中間層にすでに眠っている特徴を、短い alignment 段階で最終層に引き出す」。PE では言語アラインメント（PElang）と空間アラインメント（PEspatial、SAM 2.1 mask logits + 自己層 41 の 2 教師）の 2 種で具体化。
- [[concepts/object-detection]] — 物体検出タスク全体の俯瞰。R-CNN ファミリー（Two-stage）→ YOLO/SSD/RetinaNet（Single-stage anchor-based）→ CenterNet/FCOS（Anchor-free）→ **DETR（集合予測パラダイム）** → Promptable/Foundation model 時代（SAM/SAM 3/PE）という系譜整理。DETR ファミリー（Deformable DETR, DINO-detector, DETA, CoDETR）も網羅。

## Entities

### Models

- [[entities/dino]] — DINO 手法のスペック・主要結果・系譜（DINOv2/DINOv3 等への発展）
- [[entities/dinov2]] — DINOv2 モデル群（ViT-S/B/L/g, LVD-142M で学習, ViT-g からの蒸留）
- [[entities/dinov3]] — DINOv3 モデル群（ViT-S〜7B + ConvNeXt + dino.txt, LVD-1689M で学習, Gram anchoring, RoPE）
- [[entities/ibot]] — iBOT（Zhou et al., 2021/ICLR 2022）。DINO + MIM の online tokenizer 統合。DINOv2/DINOv3 の損失関数の直接の源。
- [[entities/mae]] — MAE（He et al., 2021）。マスクオートエンコーダ。CV MIM の中核手法、iBOT/DINOv2/DINOv3 の MIM 損失の源流。
- [[entities/clip]] — CLIP（OpenAI, 2021）と OpenCLIP（LAION）。弱教師あり代表、DINOv2 の主要ベースライン。
- [[entities/perception-encoder]] — Meta の Perception Encoder（NeurIPS 2025）。**PEcore / PElang / PEspatial の 3 バリアント**。対比学習をスケールするだけで中間層に多目的特徴が育つことを発見、alignment tuning（[[concepts/alignment-tuning]]）で末端に引き出す。MLLM・検出・dense 予測すべてで SOTA。
- [[entities/detr]] — Facebook AI の DETR（ECCV 2020）。Transformer ベースの end-to-end 物体検出。ResNet-50/101 backbone + 6 層 encoder + 6 層 decoder + 100 個の object queries。Hungarian 二部マッチング損失で NMS 不要。DETR-DC5-R101 で COCO 44.9 AP、大物体 62.3。DETR ファミリー（Deformable DETR, DAB-DETR, DN-DETR, DINO-detector, DETA, CoDETR, MDETR）の祖。
- [[entities/dino-detector]] — IDEA/HKUST/清華大の DINO 検出器（ICLR 2023）。**DETR ファミリーの集大成**。ResNet-50/SwinL backbone + 6/6 層 + 900 queries + 4D anchor box queries + deformable attention。**CDN + Mixed Query Selection + Look Forward Twice** で 12 epoch 49.4 AP / 24 epoch 51.3 AP、**DINO-SwinL（218M params）で COCO test-dev 63.3 AP** — 初の end-to-end Transformer SOTA。SwinV2-G の 1/15 パラメータで上回る。**SSL の DINO（[[entities/dino]]）とは完全に別物**。
- [[entities/glip]] — Microsoft の GLIP（CVPR 2022）。Swin-T/L backbone + DyHead 検出ヘッド + BERT テキストエンコーダ + X-MHA deep fusion。**物体検出と phrase grounding の統一定式化** で region-word alignment を学習。5 つのバリアント（GLIP-T (A/B/C), GLIP-T, GLIP-L）。GLIP-L は 27M grounding データ（FourODs + GoldG + Cap24M）で COCO ゼロショット 49.8 AP / fine-tune 61.5 AP（SOTA）。**Open-vocab 検出パラダイムの祖**、Grounding DINO / SAM 3 / OWL-ViT の源流。
- [[entities/grounding-dino]] — IDEA + Microsoft の Grounding DINO（ECCV 2024）。Swin-T/L backbone + BERT + 6 層 feature enhancer + 6 層 cross-modality decoder + 900 queries。**[[entities/glip|GLIP]] × [[entities/dino-detector|DINO 検出器]] の正統な統合**（両グループの直接共同）。2 バリアント（Grounding-DINO-T 172M / -L 341M）。Grounding-DINO-L で COCO ZS 52.5 AP / FT 63.0 AP、ODinW ZS 26.1 SOTA、RefCOCO val 90.56 SOTA。**SAM 3 と MM-Grounding-DINO の直接の祖**。
- [[entities/yolo-world]] — Tencent + 華中科技大学の YOLO-World（CVPR 2024）。YOLOv8 backbone + CLIP-base text encoder + RepVL-PAN ネック。3 バリアント（YOLO-World-S 13M / -M 29M / -L 48M、re-parameterize 版）。LVIS ZS 35.4 AP at 52 FPS V100。GLIP-T (0.12 FPS) の 433× / Grounding-DINO-T (1.5 FPS) の 35× / DetCLIP-T (2.3 FPS) の 22× 速度。**エッジデバイスでの open-vocab 検出を可能にした初の本格的研究**。GLIP-L を疑似ラベル生成 teacher として活用。
- [[entities/grounding-dino-1-5]] — IDEA Research の Grounding DINO 1.5（2024 May）。**Pro と Edge の双子モデル**。Pro: ViT-L backbone + Grounding-20M で **COCO ZS 54.3 / LVIS-mv 55.7 / ODinW35 30.2 / Fine-tune 68.1 (LVIS-mv) / 70.6 (ODinW35) SOTA**。Edge: EfficientViT-L1 backbone + Efficient Feature Enhancer（P5 のみ cross-modal 融合）で **Orin NX 10.7 FPS / A100 TRT 75.2 FPS、LVIS-mv 36.2 AP**（YOLO-Worldv2-L 超え）。**精度志向（Pro）と実用志向（Edge）を 1 モデルスイートで統合**した IDEA Research の戦略的後継。
- [[entities/dino-x]] — IDEA Research の DINO-X（2024 Nov、Grounding DINO 1.5 の正統な後継）。**unified object-centric vision model**。Pro: ViT-L + CLIP text encoder + 3 プロンプト (Text/Visual/Customized) + 4 ヘッド (Box/Mask/Keypoint/Language) + **Grounding-100M**（GD 1.5 の 5×）。**LVIS-mv 59.8 / rare APr 63.3 SOTA** / Visual Genome CIDEr **201.8** / FSC147 カウント MAE 5.6 / Hand pose HInt SOTA。Edge: EfficientViT-L2 + Knowledge Distillation (Pro→Edge) + FP16 で **Orin NX 20.1 FPS、LVIS-mv 48.3 AP**（YOLO-Worldv2-L 33.0 を +15.3 AP 凌駕）。**Universal Object Prompt で prompt-free 検出**。**「Grounding DINO 系統の到達点」**かつ SAM 3 と並行進化する unified perception model。
- [[entities/siglip]] — Google の SigLIP / SigLIP 2。CLIP の sigmoid 損失改良（v1）＋ 全部入りレシピ統合（v2: LocCa decoder + 自己蒸留＋マスク予測 + ACID + 多言語 + NaFlex）。DINOv3 の主要競合（弱教師あり）。
- [[entities/sam]] — SAM（Meta, 2023）。CV における初の本格的セグメンテーション基盤モデル。ViT-B/L/H 3 種、Apache 2.0、データエンジンで 1.1B マスクを使い訓練。
- [[entities/sam-2]] — SAM 2（Meta, 2024）。SAM の動画拡張版。Hiera 画像エンコーダ + streaming memory、Hiera-T/S/B+/L 4 種、Apache 2.0。画像でも SAM v1 比 6× 高速。
- [[entities/sam-3]] — SAM 3（Meta Superintelligence Labs, 2025）。PCS タスクを導入、PE backbone + DETR detector + presence head + SAM 2 tracker。Apache 2.0 想定。SA-Co/Gold で人間性能の 74% 達成。
- [[entities/hiera]] — Hiera（Meta, 2023）。MAE 互換の階層型 ViT、shifted window や RPB を削除したシンプル設計。SAM 2 の画像エンコーダ。
- [[entities/simclr]] — SimCLR（Google Brain, 2020）。対比 SSL の標準フレームワーク。4 コンポーネント設計と体系的アブレーションで 2020 年 SOTA を確立。
- [[entities/byol]] — BYOL（DeepMind, 2020）。オンライン×ターゲット 2 ネットワーク構造 + predictor で負例なし SSL。ResNet-200(2×) で 79.6% top-1、SimSiam/DINO 系の源流。
- [[entities/mixmatch]] — MixMatch（Google Research, NeurIPS 2019）。半教師あり学習。一貫性正則化 + シャープニング + ラベルあき・なし横断 MixUp。CIFAR-10 (250 ラベル) で 11.08%（VAT 比 3.3 倍改善）。
- [[entities/fixmatch]] — FixMatch（Google Research, NeurIPS 2020）。半教師あり学習。弱→強の非対称拡張 + τ=0.95 閾値。CIFAR-10（250 ラベル）で 5.07%（MixMatch 比 2.2× 改善）。
- [[entities/flexmatch]] — FlexMatch（東工大・Microsoft, NeurIPS 2021）。半教師あり学習。CPL クラス別動的閾値。CIFAR-10（40 ラベル）で 4.97%、収束速度 FixMatch の 1/5。
- [[entities/v-pet]] — V-PET（Ohio State, NeurIPS 2025）。VFM × PEFT アンサンブル疑似ラベリングによる SeSL。閾値も MixUp も使わず、CLIP/DINOv2 × LoRA/AdaptFormer の Mean Labels アンサンブルだけで FixMatch/FlexMatch/SoftMatch を凌駕。

### Datasets

- [[entities/imagenet]] — ImageNet（特に ImageNet-1k と ImageNet-22k）。SSL の事前学習・線形評価・k-NN 評価のデファクト基盤
- [[entities/lvd-142m]] — DINOv2 用に自動キュレーションされた 142M 画像コレクション（非公開）
- [[entities/lvd-1689m]] — DINOv3 用に階層 k-means でキュレーションされた 1.689B 画像（Instagram 由来、非公開）
- [[entities/sat-493m]] — DINOv3 衛星版用の Maxar 衛星画像 493M（0.6m/pixel, 512², 非公開）
- [[entities/wit-400m]] — CLIP の訓練データ。OpenAI が web から構築した 4 億画像-テキスト対（非公開）
- [[entities/sa-1b]] — SAM の訓練データ。データエンジンで構築された 11M 画像 × 1.1B マスク（既存最大の 400 倍）。研究用途で公開。
- [[entities/sa-v]] — SAM 2 の訓練データ。50.9K 動画 × 642.6K masklet × 35.5M マスク（既存 VOS の 53 倍）。**CC by 4.0** で公開。
- [[entities/sa-co]] — SAM 3 の訓練・評価データ。SA-Co/HQ（5.2M 画像 + 4M unique NP + 52M マスク）+ SA-Co Benchmark（**207K concepts**、既存の 50 倍以上）+ SA-Co/SYN（合成 38M 句）+ SA-Co/VIDEO（52.5K 動画）。

### People

（まだありません）

### Organizations / Benchmarks

（まだありません）

## Questions

（まだありません）

---

## 略称対応表（既出のもののみ追加）

| 略称 | 展開 | 主たる解説ページ |
|---|---|---|
| ViT | Vision Transformer | [[concepts/vision-transformer]] |
| SSL | Self-Supervised Learning | [[concepts/self-supervised-learning]] |
| WSL | Weakly-Supervised Learning | [[concepts/weakly-supervised-pretraining]] |
| KD | Knowledge Distillation | [[concepts/knowledge-distillation]] |
| MIM | Masked Image Modeling | [[concepts/masked-image-modeling]] |
| MAE | Masked Autoencoder | [[entities/mae]] |
| DAE | Denoising Autoencoder | [[concepts/denoising-autoencoder]] |
| MLM | Masked Language Modeling | [[concepts/denoising-autoencoder]] |
| online tokenizer | オンライントークナイザ | [[concepts/online-tokenizer]] |
| dVAE | discrete VAE | [[concepts/online-tokenizer]]（BEiT の文脈で言及） |
| InfoNCE | Information Noise Contrastive Estimation | [[concepts/contrastive-learning]] |
| WIT | WebImageText (400M) | [[entities/wit-400m]] |
| BPE | Byte Pair Encoding | [[entities/clip]]（テキストトークナイザ） |
| zero-shot | ゼロショット（追加訓練なしの推論） | [[concepts/zero-shot-transfer]] |
| prompt engineering | プロンプトエンジニアリング | [[concepts/zero-shot-transfer]] |
| BEiT | BERT pre-training of image transformers | [[concepts/masked-image-modeling]] |
| DINO | self-DIstillation with NO labels | [[entities/dino]] |
| DINOv2 | DINO version 2 | [[entities/dinov2]] |
| DINOv3 | DINO version 3 | [[entities/dinov3]] |
| iBOT | image BERT pre-Training with Online Tokenizer | [[entities/ibot]] |
| CLIP | Contrastive Language-Image Pre-training | [[entities/clip]] |
| OpenCLIP | open-source CLIP reproduction | [[entities/clip]] |
| PE / PEcore / PElang / PEspatial | Perception Encoder（3 バリアント） | [[entities/perception-encoder]] / [[sources/perception-encoder]] |
| alignment tuning | アラインメント・チューニング（中間層特徴を最終層に引き出すファインチューニング戦略） | [[concepts/alignment-tuning]] |
| PVD | PE Video Dataset（1M 動画 + 120K 人手精錬キャプション、15K Benchmark） | [[sources/perception-encoder]] |
| LAMB | Layer-wise Adaptive Moments optimizer for Batch training（大バッチ最適化器） | [[sources/perception-encoder]] / [[entities/clip]] |
| Attention pooling | CLIP 埋め込み構築用の pool 層を cross-attention transformer に置き換えた設計 | [[sources/perception-encoder]] |
| Progressive resolution | 訓練中に解像度を段階的に上げる戦略（98 → 154 → 224 → 336 → 448） | [[sources/perception-encoder]] |
| Mask regularization | 入力バッチの 1/16 をマスクし出力で cosine 整合させる正則化（PE 流） | [[sources/perception-encoder]] |
| AIMv2 | Autoregressive Image Models v2（Apple のキャプション化型事前学習） | [[sources/perception-encoder]] / [[entities/dinov3]] |
| InternVideo2 | Shanghai AI Lab の動画ネイティブ事前学習モデル（102M 動画） | [[sources/perception-encoder]] |
| MetaCLIP | Meta による CLIP 事前学習データキュレーション手法 | [[sources/perception-encoder]] |
| SAM 2.1 mask logits | SAM 2.1 のセグメンテーション出力確率（PEspatial の教師に使用） | [[entities/perception-encoder]] |
| DETA | Detection Transformer with Assignment（PEspatial の SOTA 検出 decoder） | [[sources/perception-encoder]] |
| CoDETR | Collaborative-DETR（[Zong et al., 2023]） | [[sources/perception-encoder]] |
| layer 41 | PEcore G（50 層）の中間層番号。空間タスクで最良性能 | [[entities/perception-encoder]] |
| layerwise feature analysis | 各層の特徴量を独立評価する解析手法（PE §3 の中核手法） | [[sources/perception-encoder]] |
| DETR | DEtection TRansformer（Carion et al., ECCV 2020、集合予測パラダイムの祖） | [[sources/detr]] / [[entities/detr]] |
| set prediction | 集合予測（順序のない出力集合の予測、DETR の核心的視点） | [[sources/detr]] |
| bipartite matching | 二部マッチング（予測と ground truth の 1 対 1 対応） | [[sources/detr]] |
| Hungarian algorithm | ハンガリアン・アルゴリズム（二部マッチングを O(N³) で解く） | [[sources/detr]] |
| Hungarian loss | DETR の集合予測損失（分類 + ボックス損失をマッチング後合計） | [[sources/detr]] |
| NMS | Non-Maximum Suppression（非最大抑制、DETR で不要に） | [[sources/detr]] / [[concepts/object-detection]] |
| anchor | 物体検出での事前定義参照ボックス（DETR は anchor フリー） | [[concepts/object-detection]] |
| proposal | 物体候補ボックス（Two-stage 検出器の第 1 段階） | [[concepts/object-detection]] |
| object queries | DETR decoder の N 個の学習可能位置埋め込み | [[sources/detr]] |
| GIoU | Generalized IoU（スケール不変ボックス類似度、DETR の核） | [[sources/detr]] |
| DC5 | Dilated C5（ResNet 最終ステージの dilation で解像度倍化） | [[sources/detr]] |
| Faster R-CNN | Two-stage 物体検出の標準（DETR の主要比較相手） | [[concepts/object-detection]] |
| Mask R-CNN | Faster R-CNN + マスク branch（インスタンスセグメンテーション） | [[concepts/object-detection]] |
| FPN | Feature Pyramid Network（Multi-scale 特徴、DETR の小物体弱点を救う系譜） | [[concepts/object-detection]] |
| Panoptic Quality (PQ) | パノプティックセグメンテーション標準指標（SQ × RQ） | [[sources/detr]] |
| things / stuff | 可算物体（人・車）/ 不可算領域（空・道路）の区別 | [[sources/detr]] |
| AP_S / AP_M / AP_L | Small / Medium / Large 物体に対する AP（COCO 標準） | [[sources/detr]] / [[concepts/object-detection]] |
| Deformable DETR | DETR の sparse attention 改善版（[Zhu et al., ICLR 2021]） | [[entities/detr]] / [[concepts/object-detection]] |
| DINO-detector | DETR ファミリー集大成（[Zhang et al., ICLR 2023]、**SSL の DINO とは別物**、slug `dino-detector` で区別） | [[sources/dino-detector]] / [[entities/dino-detector]] |
| CDN | Contrastive De-Noising（DINO 検出器の主要貢献 1、正例 + 負例の denoising 訓練） | [[sources/dino-detector]] |
| Mixed Query Selection | DINO 検出器の主要貢献 2（位置クエリのみ encoder から、コンテンツは学習可能） | [[sources/dino-detector]] |
| Look Forward Twice (LFT) | DINO 検出器の主要貢献 3（後段層の勾配で前段の box 予測を補正） | [[sources/dino-detector]] |
| Look Forward Once (LFO) | Deformable DETR の box 更新方式（勾配 detach で訓練安定化、DINO で Twice に拡張） | [[sources/dino-detector]] |
| DAB-DETR | Dynamic Anchor Box DETR（[Liu et al., ICLR 2022]）、queries を 4D anchor box として解釈 | [[entities/detr]] / [[sources/dino-detector]] |
| DN-DETR | DeNoising DETR（[Li et al., CVPR 2022]）、ノイズ付き GT を補助訓練に使い収束加速 | [[entities/detr]] / [[sources/dino-detector]] |
| dynamic anchor box | decoder で層ごとに動的更新される 4D box（DAB-DETR の核） | [[sources/dino-detector]] |
| denoising training | ノイズ加えた GT を補助訓練に使う（DN-DETR の核、DINO で contrastive 化） | [[sources/dino-detector]] |
| iterative box refinement | 各 decoder 層で box を段階的に精錬（Deformable DETR） | [[sources/dino-detector]] |
| query selection | encoder の top-K 特徴で decoder queries を初期化（Deformable DETR の two-stage） | [[sources/dino-detector]] |
| ATD(k) | Average Top-K Distance（DINO 検出器が導入した anchor-GT 距離指標） | [[sources/dino-detector]] |
| Objects365 / O365 | 365 クラスの大規模物体検出データセット（1.7M アノテーション画像） | [[sources/dino-detector]] |
| SwinL / SwinV2-G | Swin Transformer Large / V2-Giant（DINO 検出器の主要 backbone 比較） | [[sources/dino-detector]] |
| Florence | Microsoft の VLM foundation model（900M 画像-テキスト事前学習） | [[sources/dino-detector]] |
| Grounding DINO | DINO 検出器の open-vocab 拡張（[Liu et al., ECCV 2024]）、GLIP × DINO 検出器 | [[sources/grounding-dino]] / [[entities/grounding-dino]] |
| tight fusion | 浅い late-fusion ではなく複数フェーズで早期から深く融合する戦略（Grounding DINO の核） | [[sources/grounding-dino]] |
| 3-phase fusion | neck (A) + query init (B) + head (C) の 3 フェーズでの融合（Grounding DINO の設計原則） | [[sources/grounding-dino]] |
| Language-Guided Query Selection | テキスト関連性で top-K 画像 region を選び decoder queries に（Grounding DINO の Phase B） | [[sources/grounding-dino]] |
| Cross-Modality Decoder | DINO 検出器の decoder に text cross-attn を追加（Grounding DINO の Phase C） | [[sources/grounding-dino]] |
| Sub-Sentence Level Text Representation | attention mask でカテゴリ独立性 + 単語粒度を両得（Grounding DINO） | [[sources/grounding-dino]] |
| Word-level Text Representation | GLIP の方式、全カテゴリを 1 つの BERT forward でエンコード | [[sources/grounding-dino]] / [[sources/glip]] |
| Sentence-level Text Representation | 各 phrase を独立にエンコード（DetCLIP, OWL-ViT） | [[sources/grounding-dino]] |
| REC | Referring Expression Comprehension（指示表現理解）、RefCOCO/+/g | [[sources/grounding-dino]] |
| RefCOCO / RefCOCO+ / RefCOCOg | REC 標準ベンチマーク（COCO に追加注釈） | [[sources/grounding-dino]] |
| RefC | RefCOCO + RefCOCO+ + RefCOCOg の総称 | [[sources/grounding-dino]] |
| open-set object detection | 訓練に含まれない novel カテゴリを言語で指定して検出するタスク | [[sources/grounding-dino]] |
| closed-set object detection | 事前定義された固定カテゴリ集合のみを検出する古典的タスク | [[sources/grounding-dino]] / [[concepts/object-detection]] |
| OV-DETR | Open-Vocabulary DETR（CLIP queries を decoder 入力に）、Phase B のみの融合 | [[sources/grounding-dino]] |
| GLIPv2 | GLIP + segmentation 拡張（Zhang et al., NeurIPS 2022） | [[sources/grounding-dino]] |
| OmDet | Sparse R-CNN-based open-vocab 検出 | [[sources/grounding-dino]] |
| MM-Grounding-DINO | Grounding DINO のマルチモーダル拡張（OpenMMLab, 2024） | [[entities/grounding-dino]] |
| DINO-X | Grounding DINO 後継（IDEA, 2024） | [[entities/grounding-dino]] |
| YOLO-World | YOLO + open-vocabulary 検出（Cheng et al., CVPR 2024）、リアルタイム open-vocab 検出器の祖 | [[sources/yolo-world]] / [[entities/yolo-world]] |
| RepVL-PAN | Re-parameterizable Vision-Language PAN（YOLO-World のネック設計、推論時に重みに再パラメータ化可能） | [[sources/yolo-world]] |
| T-CSPLayer | Text-guided CSPLayer、max-sigmoid attention でテキストを画像特徴に注入（YOLO-World） | [[sources/yolo-world]] |
| I-Pooling Attention | Image-Pooling Attention、27 patch tokens で画像をテキスト埋め込みに注入（YOLO-World） | [[sources/yolo-world]] |
| max-sigmoid attention | softmax の代わりに max + sigmoid を使う attention（YOLO-World T-CSPLayer の核心） | [[sources/yolo-world]] |
| Prompt-Then-Detect | テキスト encoder を推論時に削除しテキスト埋め込みをモデル重みに re-parameterize するパラダイム（YOLO-World） | [[sources/yolo-world]] |
| online vocabulary | 訓練時、各 mosaic サンプルに動的に構築（最大 80 名詞、YOLO-World） | [[sources/yolo-world]] |
| offline vocabulary | 推論時、事前計算されたテキスト埋め込みセット（YOLO-World） | [[sources/yolo-world]] |
| re-parameterization | 訓練時と推論時で異なる計算グラフを使う技法（[Ding et al., RepVGG, 2021]） | [[sources/yolo-world]] |
| YOLOv8 | Ultralytics 版 YOLO（2023）、YOLO-World の base architecture | [[entities/yolo-world]] |
| CSPLayer / C2f | Cross-Stage Partial Layer（YOLOv8 の主要ブロック、YOLO-World で T-CSPLayer に拡張） | [[entities/yolo-world]] |
| PAN | Path Aggregation Network（[Liu et al., 2018]）、検出器のネック構造 | [[sources/yolo-world]] |
| Darknet | YOLO 系の標準 backbone | [[entities/yolo-world]] |
| Region-Text Contrastive Loss | 検出 + grounding + image-text データを region-text ペアで統一する損失（YOLO-World） | [[sources/yolo-world]] |
| DFL | Distribution Focal Loss（[Li et al., 2020]、YOLOv8 系の bbox 回帰用） | [[sources/yolo-world]] |
| TOOD / task-aligned label assignment | [Feng et al., 2021]、YOLO の正解割当戦略 | [[sources/yolo-world]] |
| Fixed AP | LVIS 評価の公平比較プロトコル（[Dave et al., 2021]） | [[sources/yolo-world]] |
| OVIS | Open-Vocabulary Instance Segmentation | [[sources/yolo-world]] |
| DetCLIP | ATSS-based open-vocab 検出（large-scale captioning、Yao et al., NeurIPS 2022） | [[sources/yolo-world]] / [[sources/grounding-dino]] |
| ZSD-YOLO | YOLO 版 zero-shot detection（YOLO-World の前駆、性能は劣る） | [[sources/yolo-world]] |
| MMYOLO / MMDetection | OpenMMLab のフレームワーク | [[sources/yolo-world]] |
| ViLD | Vision-and-Language knowledge Distillation（[Gu et al., ICLR 2022]、CLIP を two-stage 検出器に蒸留） | [[sources/yolo-world]] / [[sources/glip]] |
| TensorRT | NVIDIA の推論高速化ツール | [[sources/yolo-world]] / [[sources/grounding-dino-1-5]] |
| Grounding DINO 1.5 | Grounding DINO の Pro/Edge 双子拡張（IDEA Research, 2024 May） | [[sources/grounding-dino-1-5]] / [[entities/grounding-dino-1-5]] |
| Grounding DINO 1.5 Pro | ViT-L + Grounding-20M、精度志向（COCO ZS 54.3 / LVIS-mv 55.7） | [[entities/grounding-dino-1-5]] |
| Grounding DINO 1.5 Edge | EfficientViT-L1 + Efficient Feature Enhancer、エッジ展開（Orin NX 10.7 FPS） | [[entities/grounding-dino-1-5]] |
| Grounding-20M | IDEA Research が構築した 20M+ grounding 画像データセット | [[sources/grounding-dino-1-5]] |
| Efficient Feature Enhancer | Grounding DINO 1.5 Edge の新ネック（P5 のみ cross-modal 融合 + vanilla self-attn + cross-scale fusion） | [[sources/grounding-dino-1-5]] |
| EfficientViT-L1 | [Cai et al., 2023] の高速 ViT、Grounding DINO 1.5 Edge の backbone | [[sources/grounding-dino-1-5]] |
| early fusion | vision-language を encoder 段階で融合（再現率高い、hallucination 多い） | [[sources/grounding-dino-1-5]] |
| late fusion | vision-language を loss 段階のみで融合（hallucination 低い、再現率低い） | [[sources/grounding-dino-1-5]] |
| hallucination（検出） | 画像に存在しない物体を予測する誤検出 | [[sources/grounding-dino-1-5]] |
| negative sampling（検出） | 訓練時に存在しない名詞句で「何も予測しない」ことを学ばせる戦略 | [[sources/grounding-dino-1-5]] |
| Lite-DETR | [Li et al., 2023] の効率的 DETR、低レベル特徴の有用性が低いことを実証 | [[sources/grounding-dino-1-5]] |
| Cross-scale feature fusion | 異なる解像度の特徴を統合するモジュール（[Zhao et al., RT-DETR, 2023]） | [[sources/grounding-dino-1-5]] |
| NVIDIA Orin NX | NVIDIA のエッジ GPU（ロボティクス・自律走行用） | [[sources/grounding-dino-1-5]] |
| language cache | テキスト encoder 出力をキャッシュして毎回再計算しない技法 | [[sources/grounding-dino-1-5]] |
| ODinW35 | Object Detection in the Wild の 35 データセット拡張版（Grounding DINO 1.5 で導入） | [[sources/grounding-dino-1-5]] |
| DetCLIPv2 / DetCLIPv3 | DetCLIP 系列の後継、Swin-L ベースの強力な open-vocab 検出器 | [[sources/grounding-dino-1-5]] |
| T-Rex2 | visual prompt + text prompt のハイブリッド検出器（[Jiang et al., 2024]） | [[sources/grounding-dino-1-5]] |
| APE | Aligning and Prompting Everything（[Shen et al., CVPR 2024]） | [[sources/grounding-dino-1-5]] |
| GLEE-Pro | [Wu et al., CVPR 2024]、segmentation 統合モデル | [[sources/grounding-dino-1-5]] |
| OmDet-Turbo | language cache を使う高速 open-vocab 検出器（Zhao et al., 2024） | [[sources/grounding-dino-1-5]] |
| OWL-ST | OWL-ViT の self-training 拡張 | [[sources/grounding-dino-1-5]] |
| MQ-GLIP | Multi-Query GLIP（[Xu et al., NeurIPS 2023]） | [[sources/grounding-dino-1-5]] |
| V3Det | [Wang et al., ICCV 2023]、超大規模 vocabulary 検出データセット | [[sources/grounding-dino-1-5]] |
| GranuCap50M | DetCLIPv3 の grounding データセット | [[sources/grounding-dino-1-5]] |
| WebLI | Google の Web Large-scale Images-text（OWL-ST が使用） | [[sources/grounding-dino-1-5]] |
| Cap24M | GLIP の web pseudo-label data（CC3M+SBU） | [[sources/grounding-dino-1-5]] |
| fixed AP | [Dave et al., 2021]、LVIS 評価の公平比較プロトコル | [[sources/grounding-dino-1-5]] / [[sources/yolo-world]] |
| API mode | モデル重みは API 経由のみで公開（Grounding DINO 1.5 / DINO-X の特徴） | [[sources/grounding-dino-1-5]] / [[sources/dino-x]] |
| DINO-X | IDEA Research の unified object-centric vision model（2024 Nov、GD 1.5 の正統な後継） | [[sources/dino-x]] / [[entities/dino-x]] |
| DINO-X Pro | ViT-L + 3 プロンプト + 4 ヘッド + Grounding-100M（LVIS rare APr 63.3 SOTA） | [[entities/dino-x]] |
| DINO-X Edge | EfficientViT-L2 + Knowledge Distillation + FP16（Orin NX 20.1 FPS、LVIS-mv 48.3） | [[entities/dino-x]] |
| Grounding-100M | IDEA Research の 100M+ grounding 画像（Grounding-20M の 5×） | [[sources/dino-x]] |
| Grounding DINO 1.6 | GD 1.5 と DINO-X の中間版、API のみで公開（IDEA Research） | [[sources/dino-x]] |
| Universal Object Prompt | DINO-X の prompt-free 検出を可能にする事前学習済み embedding | [[sources/dino-x]] |
| prompt-free detection | ユーザー入力なしで全物体検出（DINO-X の新タスク） | [[sources/dino-x]] |
| prompt-tuning | カスタマイズ可能な prompt embedding の学習手法（[Lester et al., 2021]） | [[sources/dino-x]] |
| Customized Prompt | DINO-X の 3 種プロンプトの一つ、prompt-tuning 可能 | [[sources/dino-x]] |
| Visual Prompt | DINO-X の 3 種プロンプトの一つ、box/point ベース（T-Rex2 流） | [[sources/dino-x]] |
| object tokens | DINO-X language head で RoIAlign で抽出された領域特徴トークン | [[sources/dino-x]] |
| task tokens | DINO-X language head のタスク識別子トークン | [[sources/dino-x]] |
| RoIAlign | 領域特徴抽出演算子（Mask R-CNN 由来） | [[sources/dino-x]] |
| Mask2Former | 統一マスクアーキテクチャ（Cheng et al., CVPR 2022）、DINO-X mask head の元 | [[sources/dino-x]] |
| Mask DINO | DINO 検出器のマスク拡張（Li et al., CVPR 2023）、DINO-X mask head の元 | [[sources/dino-x]] |
| ED-Pose | End-to-end Detection Pose（Yang et al., 2023）、DINO-X keypoint head の元 | [[sources/dino-x]] |
| OPT-125M | Meta の軽量言語モデル（Zhang et al., 2022）、DINO-X language head の decoder | [[sources/dino-x]] |
| feature-based distillation | 中間特徴を Pro と Edge で整列する KD（DINO-X Edge） | [[sources/dino-x]] |
| response-based distillation | 予測 logits を Pro と Edge で整列する KD（DINO-X Edge） | [[sources/dino-x]] |
| FP16 量子化 | 浮動小数点乗算の正規化で精度保持して FP16 推論（DINO-X Edge） | [[sources/dino-x]] |
| EfficientViT-L2 | EfficientViT の強化版（DINO-X Edge backbone、GD 1.5 Edge の L1 から強化） | [[sources/dino-x]] |
| T-Rex2 | visual prompt 検出の祖（Jiang et al., 2024）、DINO-X の visual prompt encoder の元 | [[sources/dino-x]] |
| Grounded SAM | Grounding DINO + SAM のパイプライン、DINO-X が単一モデル化 | [[sources/dino-x]] |
| Grounded SAM 2 | Grounded SAM + SAM 2 の動画版 | [[sources/dino-x]] |
| Osprey | ConvNeXt-L + Vicuna-7B の region understanding モデル、DINO-X の比較相手 | [[sources/dino-x]] |
| GPT4RoI / Shikra / Ferret / Kosmos-2 / AlphaCLIP / GRIT | 関連 region MLLM | [[sources/dino-x]] |
| RefCOCOg / PACO | 領域 / part 認識データセット | [[sources/dino-x]] |
| Semantic Similarity (SS) / S-IoU | vocabulary-free 物体認識評価指標 | [[sources/dino-x]] |
| CIDEr / METEOR | 領域キャプション評価指標 | [[sources/dino-x]] |
| FSC147 / FSCD-LVIS | few-shot counting ベンチマーク | [[sources/dino-x]] |
| HInt | Hand In-the-wild ベンチマーク | [[sources/dino-x]] |
| PCK@0.05 | Percentage of Correctly Localized Keypoints @ 5% box size | [[sources/dino-x]] |
| OKS | Object Keypoint Similarity（COCO 標準） | [[sources/dino-x]] |
| CrowdPose / Human-Art | 困難な姿勢推定ベンチマーク（混雑、芸術的表現） | [[sources/dino-x]] |
| HaMeR / Hamba | hand pose の SOTA 手法 | [[sources/dino-x]] |
| GLIP | Grounded Language-Image Pre-training（Li et al., CVPR 2022）、open-vocab 検出パラダイムの祖 | [[sources/glip]] / [[entities/glip]] |
| phrase grounding | 文中の phrase と画像内の物体/領域の対応を見つけるタスク（GLIP の核） | [[sources/glip]] |
| region-word alignment | 画像 region と単語 token の埋め込み内積（GLIP の中核） | [[sources/glip]] |
| X-MHA | Cross-Modality Multi-Head Attention（GLIP の deep fusion の核） | [[sources/glip]] |
| late fusion | 最後の層だけで vision-language を融合（CLIP, ALIGN） | [[sources/glip]] |
| deep fusion | 複数層で vision-language を相互融合（GLIP, MDETR） | [[sources/glip]] |
| DyHead | Dynamic Head（[Dai et al., CVPR 2021]）、GLIP の検出 backbone | [[entities/glip]] |
| MDETR | Modulated DETR（[Kamath et al., ICCV 2021]）、テキスト条件付き DETR、GLIP の先駆 | [[sources/glip]] / [[entities/dino-detector]] |
| GoldG | MDETR キュレーションの 0.8M 人手注釈 grounding データ（Flickr30K + VG + GQA） | [[sources/glip]] |
| Cap4M / Cap24M | GLIP が生成した box 付き web 画像-テキストペア（4M / 24M） | [[entities/glip]] |
| CC / CC12M | Conceptual Captions（3M / 12M ペア、Google） | [[entities/glip]] |
| SBU | SBU Captions（1M ペア、Ordonez et al., 2011） | [[entities/glip]] |
| FourODs | 4 つの検出データセット結合（O365 + OI + VG + ImageNetBoxes、2.66M） | [[entities/glip]] |
| ODinW | Object Detection in the Wild（GLIP が導入した 13 データセットベンチマーク） | [[sources/glip]] |
| EgoHands / Pothole / ThermalDogsandPeople | ODinW の代表例（自己中心 / 道路穴 / 赤外線） | [[sources/glip]] |
| Flickr30K Entities | phrase grounding 標準ベンチマーク（Plummer et al., 2015） | [[sources/glip]] |
| Visual Genome (VG) | 110,689 ユニーク phrase を持つ大規模グラフベース画像注釈 | [[sources/glip]] |
| GQA | Question Answering ベースの visual reasoning データセット | [[sources/glip]] |
| LVIS APr / APc / APf | rare / common / frequent カテゴリの AP（LVIS の長尾分布対応） | [[sources/glip]] |
| auxiliary decoding losses | 各 decoder 層の出力に同じ Hungarian 損失を適用する DETR の安定化技法 | [[sources/detr]] |
| AdamW | Adam + decoupled weight decay（[Loshchilov & Hutter, ICLR 2019]） | [[sources/detr]] |
| Xavier init | 重みを fan-in/out で正規化する初期化（[Glorot & Bengio, 2010]） | [[sources/detr]] |
| Focal loss | クラス不均衡用の重み付き CE（[Lin et al., 2017]、RetinaNet/SAM も採用） | [[concepts/object-detection]] |
| SigLIP / SigLIP 2 | Sigmoid Loss Image-Language Pre-training | [[entities/siglip]] |
| AM-RADIO | Agglomerative Models — RADIO | [[entities/dinov3]]（短い言及） |
| EVA-CLIP | enhanced CLIP scaling | [[entities/dinov3]]（短い言及） |
| Franca | open-data SSL model | [[entities/dinov3]]（短い言及） |
| Web-DINO | web-scale DINO | [[entities/dinov3]]（短い言及） |
| V-JEPA / V-JEPA 2 | Video JEPA | [[entities/dinov3]]（短い言及） |
| JEPA | Joint-Embedding Predictive Architecture | [[entities/dinov3]]（短い言及） |
| SAM | Segment Anything Model | [[entities/sam]] |
| SAM 2 | Segment Anything Model 2（動画版） | [[entities/sam-2]] |
| SAM 3 | Segment Anything Model 3（コンセプト版） | [[entities/sam-3]] |
| SA-1B | Segment Anything 1 Billion (dataset) | [[entities/sa-1b]] |
| SA-V | Segment Anything Video (dataset) | [[entities/sa-v]] |
| SA-Co | Segment Anything with Concepts (dataset/benchmark) | [[entities/sa-co]] |
| Hiera | Hierarchical Vision Transformer | [[entities/hiera]] |
| PVS | Promptable Visual Segmentation（SAM 1/2 のタスク） | [[concepts/promptable-segmentation]] |
| PCS | Promptable Concept Segmentation（SAM 3 のタスク） | [[concepts/promptable-concept-segmentation]] |
| promptable segmentation | プロンプト可能セグメンテーション | [[concepts/promptable-segmentation]] |
| masklet | 動画全体の時空間マスク | [[entities/sa-v]] / [[entities/sam-2]] |
| NP | Noun Phrase（短い名詞句、SAM 3 のテキストプロンプト形式） | [[concepts/promptable-concept-segmentation]] |
| hard negative | 敵対的に困難な negative ラベル | [[entities/sa-co]] |
| exemplar | 画像 exemplar（SAM 3 の画像プロンプト） | [[entities/sam-3]] |
| presence token / head | SAM 3 の認識と位置特定を分離するトークン | [[entities/sam-3]] |
| AI verifier | Llama 3.2 でファインチューンされた自動検証者 | [[entities/sa-co]] |
| MV / EV | Mask Verification / Exhaustivity Verification | [[entities/sa-co]] |
| cgF₁ | classification-gated F1（SAM 3 主要メトリクス） | [[sources/sam-3]] |
| pmF₁ | positive media-phrase F1 | [[sources/sam-3]] |
| IL_MCC | Image-Level Matthews Correlation Coefficient | [[sources/sam-3]] |
| pHOTA | positive HOTA（動画 PCS の主要メトリクス） | [[sources/sam-3]] |
| HOTA | Higher Order Tracking Accuracy | [[sources/sam-3]] |
| DETR | DEtection TRansformer | [[entities/sam-3]] |
| MDETR | Modulated DETR（テキスト条件付き DETR） | [[entities/sam-3]] |
| MaskFormer | マスクヘッドアーキテクチャ | [[entities/sam-3]] |
| SAM 3 Agent | SAM 3 を MLLM ツールとして使うパターン | [[entities/sam-3]] |
| SigLIP | Sigmoid Loss for Language-Image Pre-training | [[entities/siglip]] |
| SigLiT | Sigmoid LiT（frozen 画像 backbone 版 SigLIP） | [[entities/siglip]] |
| mSigLIP | multilingual SigLIP（100 言語版） | [[entities/siglip]] |
| sigmoid loss | ペア独立の二値分類損失（SigLIP の核心） | [[sources/siglip]] / [[concepts/contrastive-learning]] |
| WebLI | Google 内部の Web 由来画像-テキストデータ | [[entities/siglip]]（PaLI 由来） |
| SO-400M | Shape-Optimized 400M ViT | [[entities/siglip]] |
| XM3600 | Crossmodal-3600（36 言語の多言語検索ベンチマーク） | [[entities/siglip]] |
| LiT | Locked-image Tuning（frozen backbone + テキスト訓練） | [[entities/siglip]]（SigLiT の元） |
| InfoNCE | Info Noise Contrastive Estimation（softmax 対比損失） | [[concepts/contrastive-learning]] |
| big_vision | Google の公開コードベース | [[entities/siglip]] |
| β₂ | Adam/AdaFactor のモメンタム係数（SigLIP では 0.95） | [[entities/siglip]] |
| chunked implementation | デバイス間で per-device $b^2$ メモリの効率実装 | [[entities/siglip]] |
| SigLIP 2 | SigLIP version 2（全部入り統合レシピ） | [[entities/siglip]] / [[sources/siglip-2]] |
| NaFlex | Native aspect ratio + Flexible sequence length バリアント | [[entities/siglip]] / [[sources/siglip-2]] |
| NaViT | Native-resolution ViT（[Dehghani et al. 2024]） | [[sources/siglip-2]]（言及） |
| FlexiViT | Flexible patch-size ViT（[Beyer et al. 2023]） | [[sources/siglip-2]]（言及） |
| LocCa | Location-aware Captioner、decoder ベース事前学習レシピ | [[sources/siglip-2]] / [[entities/siglip]] |
| SILC | Self-supervised + Image-Language Contrastive learning | [[sources/siglip-2]] |
| TIPS | Text-Image Pretraining with Spatial awareness | [[sources/siglip-2]] |
| ACID | Active Curation Implicit Distillation | [[sources/siglip-2]] / [[concepts/knowledge-distillation]] |
| ACED | ACID + Explicit Distillation | [[sources/siglip-2]] / [[concepts/knowledge-distillation]] |
| MAP head | Multi-head Attention Pooling | [[entities/siglip]] |
| Gemma / Gemma 2 | Google の LLM ファミリー、SigLIP 2 のトークナイザに採用 | [[entities/siglip]] |
| PaliGemma / PaliGemma 2 | SigLIP + Gemma の VLM | [[entities/siglip]] |
| representation bias | ランダムオブジェクトと性別の偏った関連付け度 | [[sources/siglip-2]] |
| GeoDE | Geographically Diverse Evaluation データセット | [[sources/siglip-2]] |
| Dollar Street | 所得別多様性評価データセット | [[sources/siglip-2]] |
| RefCOCO / RefCOCO+ / RefCOCOg | 参照表現理解ベンチマーク | [[sources/siglip-2]] |
| DPT | Dense Prediction Transformer | [[sources/siglip-2]] |
| OWL-ViT | Open-vocab detection 手法 | [[sources/siglip-2]] |
| Cat-Seg | Open-vocab segmentation 手法 | [[sources/siglip-2]] |
| HierText / SciCap / Screen2Words / TextCaps | OCR/文書/スクリーン系ベンチマーク | [[sources/siglip-2]] |
| TPUv5e | Google 第 5 世代 TPU | [[sources/siglip-2]]（最大 2048 チップ） |
| VOS | Video Object Segmentation | [[entities/sam-2]] |
| iVOS | interactive VOS | [[entities/sam-2]] |
| 𝒥&ℱ | VOS 標準精度指標（領域類似度 𝒥 + 境界精度 ℱ） | [[sources/sam-2]] |
| 2D-RoPE | 2 次元 Rotary Positional Embedding | [[concepts/rotary-position-embeddings]] |
| FlashAttention | 効率的 attention カーネル | [[entities/sam-2]] |
| MOSE | 困難 VOS ベンチマーク | [[entities/sam-2]] |
| LVOS / LVOSv2 | 長期 VOS ベンチマーク | [[entities/sam-2]] |
| DAVIS | 古典的 VOS ベンチマーク | [[entities/sam-2]] |
| YouTube-VOS / YTVOS | 大規模 VOS ベンチマーク | [[entities/sam-2]] |
| XMem / XMem++ | 強力な対話的 VOS ベースライン | [[entities/sam-2]] |
| Cutie | 強力な対話的 VOS ベースライン | [[entities/sam-2]] |
| FPN | Feature Pyramid Network | [[entities/hiera]] |
| GRU | Gated Recurrent Unit（SAM 2 で試したが不採用） | [[entities/sam-2]] |
| RITM | Reviving Iterative Training with Mask guidance | [[sources/segment-anything]]（対話的セグメンテーション主要ベースライン） |
| ViTDet | Vision Transformer Detector | [[sources/segment-anything]]（インスタンスセグメンテーションのボックス供給元） |
| LVIS | Large Vocabulary Instance Segmentation | [[sources/segment-anything]] |
| COCO | Common Objects in Context | [[sources/segment-anything]] |
| NMS | Non-Maximum Suppression | [[sources/segment-anything]] |
| IoU / mIoU | Intersection over Union / mean IoU | [[sources/segment-anything]] |
| focal loss | 不均衡分類用損失 | [[sources/segment-anything]] |
| dice loss | セグメンテーション用領域重なり損失 | [[sources/segment-anything]] |
| MIAP | More Inclusive Annotations for People | [[sources/segment-anything]]（公平性評価データセット） |
| RAI | Responsible AI | [[sources/segment-anything]] |
| stuff / things | 不可算/可算オブジェクト区別 | [[sources/segment-anything]] |
| modal / amodal mask | 可視部分のみ/遮蔽部含むマスク | [[sources/segment-anything]] |
| BSDS500 | Berkeley Segmentation Dataset 500 | [[sources/segment-anything]]（エッジ検出ベンチマーク） |
| ODS / OIS / AP / R50 | エッジ検出標準指標 | [[sources/segment-anything]] |
| VGGT | Visual Geometry Grounded Transformer | [[entities/dinov3]] |
| DAv2 | Depth Anything V2 | [[entities/dinov3]] |
| SimCLR | A Simple Framework for Contrastive Learning of Visual Representations | [[entities/simclr]] |
| NT-Xent | Normalized Temperature-scaled Cross Entropy（SimCLR の損失関数, InfoNCE の実装） | [[sources/simclr]] |
| LARS | Layer-wise Adaptive Rate Scaling（大バッチ学習用オプティマイザ） | [[sources/simclr]] |
| MoCo | Momentum Contrast（momentum encoder + memory bank 版対比 SSL） | [[concepts/contrastive-learning]] |
| PIRL | Pretext-Invariant Representations Learning | [[concepts/contrastive-learning]] |
| Global BN | Global Batch Normalization（分散訓練での情報リーク防止） | [[sources/simclr]] |
| BYOL | Bootstrap Your Own Latent（負例なし SSL の先駆け） | [[entities/byol]] |
| predictor | BYOL のオンライン側追加 MLP（崩壊防止の鍵） | [[sources/byol]] |
| MixMatch | 半教師あり学習アルゴリズム（一貫性正則化 + シャープニング + MixUp 統合） | [[entities/mixmatch]] |
| MixUp | 2 例を凸補間する正則化手法（Zhang et al., 2017） | [[sources/mixmatch]] |
| Label Guessing | K 回拡張の予測平均で疑似ラベルを生成 | [[sources/mixmatch]] |
| Sharpening（半教師あり）| 温度で確率分布を先鋭化してエントロピーを下げる操作 | [[sources/mixmatch]] |
| SeSL / semi-SSL | Semi-Supervised Learning（半教師あり学習） | [[concepts/semi-supervised-learning]] |
| Mean Teacher | EMA ターゲット教師による一貫性正則化 | [[concepts/semi-supervised-learning]] |
| VAT | Virtual Adversarial Training（仮想敵対的訓練） | [[concepts/semi-supervised-learning]] |
| FixMatch | 高信頼閾値 τ=0.95 + 弱→強拡張での半教師あり学習（MixMatch 後継） | [[entities/fixmatch]] |
| RandAugment | Randomly-selected Augmentation（事前探索なしに M 種の拡張を適用） | [[sources/fixmatch]] |
| CTAugment | Control Theory Augment（ラベルあきデータで拡張強度をオンライン調整） | [[sources/fixmatch]] |
| Cutout | 画像の矩形領域をゼロでマスクする正則化 | [[sources/fixmatch]] |
| confirmation bias / 確証バイアス | 誤疑似ラベルが自己強化されるループ | [[sources/fixmatch]] |
| DA / Distribution Alignment | モデルの周辺予測分布をラベル分布に合わせるバイアス補正 | [[sources/fixmatch]] |
| barely supervised | 1 クラスあたり数枚程度という極端な少ラベル設定 | [[sources/fixmatch]] |
| τ (tau) | 信頼度閾値。FixMatch ではデフォルト 0.95（FlexMatch ではクラス別動的閾値の上限） | [[entities/fixmatch]] |
| FlexMatch | FixMatch にクラス別動的閾値（CPL）を導入した後継手法 | [[entities/flexmatch]] |
| CPL | Curriculum Pseudo Labeling（カリキュラム疑似ラベリング） | [[sources/flexmatch]] |
| σ_t(c) / β_t(c) | FlexMatch の推定学習効果 / 正規化学習効果（クラス別） | [[sources/flexmatch]] |
| カリキュラム学習 | 簡単→難しいサンプルの段階的導入戦略（Bengio 2009） | [[sources/flexmatch]] |
| TorchSSL | FlexMatch と同時公開された PyTorch ベース SSL 統合コードベース | [[entities/flexmatch]] |
| FreeMatch | 2 レベル自由閾値による FlexMatch の後継（Wang et al., 2023） | [[entities/flexmatch]] |
| SoftMatch | ガウス重み付きソフト閾値（FlexMatch の後継, Chen et al., 2023） | [[entities/flexmatch]] |
| VFM | Vision Foundation Model（視覚基盤モデル, CLIP/DINOv2 等） | [[concepts/foundation-model]] |
| PEFT / PETL | Parameter-Efficient (Transfer) Fine-Tuning（パラメータ効率的ファインチューニング） | [[concepts/parameter-efficient-fine-tuning]] |
| LoRA | Low-Rank Adaptation（重み更新を低ランク行列で近似） | [[concepts/parameter-efficient-fine-tuning]] |
| AdaptFormer | アダプタベース PEFT（ViT 向け bottleneck MLP） | [[concepts/parameter-efficient-fine-tuning]] |
| BitFit | bias 項のみ更新の超軽量 PEFT | [[concepts/parameter-efficient-fine-tuning]] |
| VPT | Visual Prompt Tuning（学習可能プロンプトトークン） | [[concepts/parameter-efficient-fine-tuning]] |
| ConvPass / Fact-TT | 畳み込み / テンソル分解ベース PEFT | [[concepts/parameter-efficient-fine-tuning]] |
| V-PET | VFM-PEFT Ensemble Training（VFM × PEFT アンサンブル疑似ラベリング） | [[entities/v-pet]] |
| Mean Labels | one-hot 化後の平均によるアンサンブル戦略（V-PET 提案） | [[entities/v-pet]] |
| ST / PET | Self-Training / PEFT Ensemble Training（V-PET のバリアント） | [[entities/v-pet]] |
| VTAB | Visual Task Adaptation Benchmark | [[sources/revisiting-ssl-foundation-models]] |
| DTD | Describable Textures Dataset（VTAB 内テクスチャ認識データセット） | [[sources/revisiting-ssl-foundation-models]] |
| SUN397 | Scene Understanding Database 397 クラス | [[sources/revisiting-ssl-foundation-models]] |
| RESISC45 | Remote Sensing Image Scene Classification（45 クラス） | [[sources/revisiting-ssl-foundation-models]] |
| Retinopathy | 糖尿病性網膜症データセット（医療画像 SeSL） | [[sources/revisiting-ssl-foundation-models]] |
| CLEVR-C | CLEVR Count（合成推論データセット） | [[sources/revisiting-ssl-foundation-models]] |
| AMI / ARI / V-Measure / FMI | クラスタリング評価指標（教師なしハイパラチューニング基準） | [[sources/revisiting-ssl-foundation-models]] |
| BNM | Batch Nuclear-norm Maximization（特徴行列核ノルム） | [[sources/revisiting-ssl-foundation-models]] |
| RankMe | 表現行列のランクで表現品質を測る（Garrido et al., 2023） | [[sources/revisiting-ssl-foundation-models]] |
| CHI | Calinski-Harabasz Index（クラスタ品質指標） | [[sources/revisiting-ssl-foundation-models]] |
| FineSSL | CLIP 視覚バックボーン + 平衡マージン softmax SeSL 手法（Gan & Wei, 2024） | [[sources/revisiting-ssl-foundation-models]] |
| SoftMatch | ガウス重み付け SeSL 手法（Chen et al., 2023） | [[sources/revisiting-ssl-foundation-models]] |
| Brier スコア | 多クラス予測確率と正解の L2 二乗距離（有界な損失） | [[sources/mixmatch]] |
| PATE | Private Aggregation of Teachers' Ensembles（差分プライバシー学習） | [[sources/mixmatch]] |
| Wide ResNet / WRN | Wide Residual Network（幅広い ResNet） | [[sources/mixmatch]] |
| DPT | Dense Prediction Transformer | [[entities/dinov3]] |
| LVD-142M | Large-scale Visual Dataset (142M) | [[entities/lvd-142m]] |
| LVD-1689M | Large-scale Visual Dataset (1689M) | [[entities/lvd-1689m]] |
| SAT-493M | Satellite Dataset (493M) | [[entities/sat-493m]] |
| Gram Anchoring | パッチ間 Gram 行列正則化 | [[concepts/gram-anchoring]] |
| RoPE | Rotary Position Embeddings | [[concepts/rotary-position-embeddings]] |
| register tokens | レジスタトークン | [[entities/dinov3]] |
| EMA | Exponential Moving Average | [[concepts/self-supervised-learning]] |
| k-NN | k-Nearest Neighbors classifier | [[concepts/knn-evaluation-protocol]] |
| ILSVRC | ImageNet Large Scale Visual Recognition Challenge | [[entities/imagenet]] |
| SK | Sinkhorn-Knopp | [[entities/dinov2]] |
| KoLeo | Kozachenko-Leonenko regularizer | [[entities/dinov2]] |
| SwiGLU | Swish-Gated Linear Unit | [[entities/dinov2]] |
| FFN | Feed-Forward Network | [[entities/dinov2]] |
| FSDP | Fully-Sharded Data Parallel | [[entities/dinov2]] |
| DDP | Distributed Data Parallel | [[entities/dinov2]] |
| FlashAttention | 高効率 attention 実装 | [[entities/dinov3]] |
| PUE | Power Usage Effectiveness | [[entities/dinov2]] |
| GLDv2 | Google Landmarks Dataset v2 | [[sources/dino-emerging-properties-in-self-supervised-vit]] |
| TTA | Test-Time Augmentation | [[sources/dinov3]] |
| OOD | Out-Of-Distribution | [[sources/dinov3]] |
| CNX / ConvNeXt | Convolutional Next | [[entities/dinov3]] |
| WRI | World Resources Institute | [[sources/dinov3]] |
| AIMv2 | Autoregressive Image Models v2 | [[sources/dinov3]] |
| LiT | Locked-image Tuning | [[entities/dinov3]] |
| dino.txt | DINO + text alignment | [[entities/dinov3]] |
| CNN / convnet | Convolutional Neural Network | （未作成） |
| MLP | Multi-Layer Perceptron | （未作成） |
| BN | Batch Normalization | （未作成） |
| CE | Cross Entropy | （未作成） |
| BYOL | Bootstrap Your Own Latent | （未作成） |
| MoCo | Momentum Contrast | （未作成） |
| SwAV | Swapping Assignments between multiple Views | （未作成） |
| BERT | Bidirectional Encoder Representations from Transformers | （未作成） |
| VLM | Vision-Language Model | （未作成） |
