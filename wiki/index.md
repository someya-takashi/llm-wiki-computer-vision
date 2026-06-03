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
- [[sources/internvl]] — InternVL（Chen et al., OpenGVLab Shanghai AI Lab, CVPR 2024）。**視覚エンコーダを 6B にスケールアップ + 8B 言語ミドルウェア QLLaMA で LLM と整列させた初の本格的視覚言語基盤モデル**。InternViT-6B + QLLaMA を **3 段階訓練（contrastive 4.98B → generative 1.03B → SFT 4M）** で接続、対比 + 生成 + 対話を 1 モデル統合。**InternVL-C**: IN-1K ZS **83.2**（EVA-02-CLIP-E+ 82.0 超え）/ 多言語 5 言語平均 **64.0** SOTA / Kinetics-700 8F **60.6**（ViCLIP +6.3）。**InternVL-G**: Flickr30K I2T R@1 **95.7** / COCO caption CIDEr **128.2 SOTA**。**InternVL-Chat-13B (QLLaMA)**: MME **1586.4**（LLaVA-1.5 +55）/ POPE 87.6 / GQA 66.6。**ViT-22B（21.7B）を 1/3.7 のパラメータで ADE20K linear probe +12.6 mIoU 上回る**（5.9B で 47.2 vs ViT-22B 34.6）。InternVL 1.5 / 2.0 / 2.5 / 3 シリーズの起点。
- [[sources/internvl-1-5]] — InternVL 1.5（Chen et al., OpenGVLab Shanghai AI Lab, 2024 April, technical report）。**「How Far Are We to GPT-4V?」**: オープンソース MLLM と商用 MLLM（GPT-4V / Gemini Pro / Claude-3 / Qwen-VL-Max）の能力差を初めて明確に縮めた論文。**InternViT-6B-448px-V1.5（45 層、dynamic 448）+ MLP プロジェクタ + InternLM2-20B-Chat = 26B**（[[entities/internvl\|InternVL 1.0]] の QLLaMA を廃止し LLaVA 系の MLP に統一）。**3 つの改善**: (1) **継続事前学習** で InternViT-6B 強化（後ろから 4 層目が MLLM タスクで最良 → 48 層 → 45 層、「中間層特徴」発見の先駆）、(2) **動的高解像度** 35 通りのアスペクト比 × 1-12 タイル（テスト時 40 タイル = 4K 解像度）+ Pixel Shuffle で visual token 1/4 圧縮、(3) **高品質バイリンガル（英中）データ** + PaddleOCR で大規模擬似ラベリング + LLM ベース翻訳パイプライン。**18 ベンチ中 8 SoTA**: **ChartQA 83.8 / OCRBench 724 / MMBench-CN 82.0 / CCBench 69.8 (GPT-4V +23.3) / HallusionBench 49.3 / MathVista 53.5 (GPT-4V +3.6)**。**Larger LLMs need Larger VFMs**（6B VFM vs 300M VFM、34B LLM）のアブレーション、**動的解像度はタスク依存**（OCR は高解像度、シーン推論は低解像度）の発見。**ConvBench マルチターンでは GPT-4V に大敗**（17.65 vs 39.51）。InternVL 2.0 / 2.5 / 3 の直接の起点、MLLM 競争の風向きを変えた論文。
- [[sources/mini-internvl]] — Mini-InternVL（Gao et al., OpenGVLab Shanghai AI Lab, 2024 Oct, technical report）。**「5% のパラメータで 90% の性能」**: InternVL シリーズの **「軽量化 + ドメイン特化」分枝**を確立した論文。**[[entities/internvit-300m\|InternViT-300M]]**（**CLIP-ViT-L-336px で初期化** + **InternViT-6B から negative cosine similarity 損失で蒸留**）+ 軽量 LLM（**Qwen2-0.5B** / **InternLM2-1.8B** / **Phi-3-Mini**）で **Mini-InternVL-1B / 2B / 4B** を構築。動的解像度・Pixel Unshuffle 等の構造は [[sources/internvl-1-5\|InternVL 1.5]] と同じ。**汎用ベンチ**: Mini-InternVL-4B (4B) で **InternVL2-Llama3-76B (76B) の 90% 性能**（Avg 72.8 vs 81.4）、Gemini-Pro-1.5 とほぼ同等。**統一ドメイン適応フレームワーク**: 5 種タスク（画像分類 / grounding / 領域知覚 / 多視点 / 動画）を VQA 形式に統一、3 ドメイン（**自律走行 / 医療 / リモートセンシング**）で Mini-InternVL-DA-1B/2B/4B を構築。**自律走行 DriveLM Challenge**: Mini-InternVL-DA-2B が **2B で 26B SOTA (InternVL4Drive-v2) に匹敵**（0.5958 vs 0.6002）。**MME-RealWorld 自律走行**: DA-4B が **GPT-4o を 24.78 ポイント圧倒**（49.38 vs 24.60）。**医療 GMAI-MMBench**: DA-4B が LLaVA-Med / RadFM / Claude3-Opus を凌駕（2D Seg/Cls/Det）。**リモセン DIOR-RSVG**: DA-4B 92.04（SkyEyeGPT 超え）。**知識蒸留アブレーション**: InternViT-300M が CLIP-ViT-L (同サイズ) を OCR で +8.4 / Chart で +5.3 / InfoVQA で +8.1 圧倒。汎用:特化 = 1:4（r=0.25）で性能ピーク、Full-parameter > Freezing ViT > LoRA。
- [[sources/internvl-2-5]] — InternVL 2.5（Chen et al., OpenGVLab Shanghai AI Lab, 2024 Dec, technical report）。**「MMMU で 70% を超えた初のオープンソース MLLM」**: MMMU 70.1%（GPT-4o 69.1 / Claude-3.5-Sonnet 68.3 / Gemini-1.5-Pro 62.2 を凌駕）。アーキテクチャは [[sources/internvl-1-5\|InternVL 1.5]] と完全に同じ ViT-MLP-LLM、**訓練戦略・データ・テスト時スケーリングのみで性能境界を拡張** という哲学。**1B/2B/4B/8B/26B/38B/78B の 7 サイズスイート** で軽量と大規模を統合。視覚: **InternViT-300M-V2.5** （1B-8B）or **InternViT-6B-V2.5**（26B+、45 層、5.5B）、LLM: **InternLM 2.5 / Qwen 2.5**。**Progressive Scaling Strategy を初めて公式化**: 小型 LLM（20B）で ViT 訓練 → 大型 LLM（72B）に転送、**Qwen2-VL の 1/12 の訓練トークン**（120B vs 1.4T）。**Test-Time Scaling**: Chain-of-Thought + Majority Voting で **MMMU +3.7 ポイント追加改善**。**データ品質**: 5.1M → 7.3M → **16.3M** サンプル、繰り返し検出 + LLM スコアリング + ヒューリスティック規則の **3 戦略フィルタリング**、CoT デッドロック問題解消。**主要結果**: MathVista **72.3**（GPT-4o 63.8 +8.5）/ MathVerse 51.7 / VCR-EN-Easy 95.7（InternVL2-2B 32.9 → 2.5-2B 93.2 +60.3 で 22K サンプル追加効果）/ RefCOCO 92.3 SOTA / Video-MME 72.1 + MVBench 76.4 + MMBench-Video 1.97 + MLVU 75.7（複数 SOTA）/ MME 2494.5 + MMB-EN/CN 88.3/88.5 + MMStar 69.5 SOTA / 純粋言語 +0.5～+1.4（vs 基盤 LLM、2.0 系は -2.1～-2.3 低下していた）。**InternViT 系統で「最終層の線形分離性が低下しつつ open-set 意味を捕捉」を独立発見**（[[concepts/alignment-tuning\|PE の中間層特徴]] と同等の現象）。**弱点**: WildVision 71.4（GPT-4o 80.6 -9.2、長応答品質）、MMVet v2 65.5（GPT-4o 71.0 / Claude-3.5 71.8）、Stage 3 post-training（DPO/RLHF）未実施。**InternVL 商用追従ラインの最終到達点**、次の InternVL 3 で「Native multimodal pretraining」へ哲学転換。
- [[sources/internvl-3]] — InternVL 3（Zhu et al., OpenGVLab Shanghai AI Lab, 2025 Apr, technical report）。**「Native Multimodal Pre-Training パラダイムを確立、MMMU 72.2 で SOTA 更新」**: InternVL 1.0-2.5 の「LLM Chat 版から MLLM を事後改造」する伝統を捨て、**Qwen2.5 base + InternViT で、テキスト + マルチモーダルデータを共同で事前学習**（言語 50B + マルチモーダル 150B、1:3 比率）。**3 つの主要技術**: (1) **Native Multimodal Pre-Training**（全層共同訓練、視覚トークンは予測しない text-only loss + square averaging）、(2) **V2PE（Variable Visual Position Encoding）** で視覚トークンに位置インクリメント δ < 1 を使用（{1, 1/2, ..., 1/256} からランダム選択、画像内で δ 一定）、長文脈対応、$\delta=1/4$ で最良性能、(3) **MPO（Mixed Preference Optimization）** = DPO + BCO + LM 損失の混合、300K 選好データで CoT 推論強化、**+4.5 ポイント改善**（38B）。**Test-Time Scaling**: **VisualPRM-8B（Visual Process Reward Model）** で各推論ステップに +/- スコアを付与、Best-of-8 で **InternVL3-1B が +9.9 ポイント** という小型モデルで顕著な改善。**1B/2B/8B/9B/14B/38B/78B の 7 サイズ**（9B のみ InternLM3-8B base、それ以外は Qwen2.5 base）。**主要結果**: MMMU **72.2 で SOTA**（GPT-4o 70.7 / Qwen2.5-VL-72B 68.2 / Gemini-2.0-Pro 69.9 超え、Claude-3.7-Sonnet 75.0 に -2.8）/ MathVista **79.0**（GPT-4o 60.0 +19.0）/ OCRBench **906 で史上初の 900 超え** / MME 2549.8 / MMB-EN/CN 89.0/88.7 / MMStar 72.5 / MVBench 78.7 + MLVU 79.5 + CG-Bench 48.4/65.3（複数 SOTA）/ **VSI-Bench 空間推論で GPT-4o (34.0) を InternVL3-8B (42.1) で +8.1 圧倒**、3D シーン理解の新水準。**「マルチモーダル化で言語能力が強くなる」初実証**: 同じ Qwen2.5 base から派生した Qwen2.5-Chat より純粋言語 17 ベンチで **+1.6〜+8.9 強化**（小型ほど顕著、1B で +8.9）。**訓練データ + モデル重みを完全公開**（open-science 強化、`OpenGVLab/InternVL-Data`）。**弱点**: Visual Grounding でシリーズ初の退行（92.3 → 91.4、grounding データ比率低下）、Claude-3.7-Sonnet MMMU 75.0 / Gemini-2.0-Pro MathVerse 67.3 にはまだ届かない。InternVL シリーズ第 7 世代、**InternVL の哲学転換を象徴する論文**。
- [[sources/mpo]] — MPO（Wang et al., OpenGVLab Shanghai AI Lab, 2024 Nov, technical report）。**「MLLM の CoT 推論で性能が悪化する」現象を初めて本格的に解決した論文**。著者らは「SFT が teacher forcing による分布シフトを生み、長い CoT rationale で誤差蓄積する」と分析。**3 点セット** を提案: (1) **Mixed Preference Optimization (MPO)** = **DPO（相対選好）+ BCO（絶対品質）+ SFT loss（生成過程）** の 3 損失混合（$w_p=0.8, w_q=0.2, w_g=1.0$、$\beta=0.1$）、(2) **MMPR データセット**（[[entities/mmpr\|3M 選好ペア、6 ドメイン × 18 データセット]]）、(3) **DropoutNTP**（画像なしで応答を補完して rejected 生成、**RLAIF-V の 57.5% コスト** で同等品質）。**主要結果**: **InternVL2-8B-MPO が MathVista +8.7 ポイント（67.0、10× 大きい InternVL2-76B 67.2 と同等）、M3CoT +19.9（59.3 → 79.2）、MathVision +5.3（25.7 で当時オープン MLLM 新 SOTA）**。**CoT が direct より良くなる**（M3CoT で +2.0、baseline は -2.3）。**テキスト専用ベンチでも改善**（MMPR にテキスト専用データなしにもかかわらず、TheoremQA +5.2 / IFEval +4.1）。**10 個の PO 手法を比較**（DPO/RSO/IPO/cDPO/RobustDPO/BCO/SPPO/AOT/TR-DPO/ORPO）し、**「DPO + BCO + SFT loss」が最良 CoT 性能** を発見。**「SFT loss が CoT 強化の鍵」「参照モデル凍結が必須」** という重要な観察。**[[entities/internvl-3\|InternVL 3]] の Stage 3 で正式採用**（MMPR v1.2 + 300K 選好ペア、+4.1〜+4.5 ポイント推論改善）、**[[entities/internvl-3-5\|InternVL 3.5]] では Cascade RL の Stage 1 として進化**（MPO + GSPO の 2 段階で SFT 単独から最大 +12 ポイント改善）。**InternVL シリーズの永続的技術となった**。「マルチモーダル選好最適化が言語能力も強化」という [[entities/internvl-3\|InternVL 3]] の「Native Pre-Training で言語が強くなる」発見の **先例**。
- [[sources/sdxl]] — SDXL（Podell et al., Stability AI Applied Research, 2023 Jul, arXiv:2307.01952）。**「wiki 初の Stable Diffusion / Latent Diffusion Model 系統、オープン text-to-image 拡散モデルを商用フロンティアと互角に押し上げた決定版」**。**Stable Diffusion 第 3 世代**（SD 1.x → SD 2.x → **SDXL**）、Rombach lab（LDM 原論文の著者ら）の正統な後継。**5 つの主要改善**: (1) **3× 大きい UNet（2.6B、SD 1.x の 860M から）**: 異種 transformer ブロック分布 **[0, 2, 10]**（最高レベルで省略、低レベルで多用）、channel mult [1, 2, 4]、最低レベル（8× ダウンサンプリング）完全削除、(2) **二重テキスト・エンコーダ**: **[[entities/clip\|CLIP]] ViT-L + OpenCLIP ViT-bigG**（817M、context dim **2048**、ペナルティメイト出力をチャンネル軸で連結）+ **OpenCLIP の pooled 埋め込み**を timestep に加算、(3) **3 つの新規条件付け技術**: **size-conditioning**（フーリエ特徴で元の H, W を埋め込み → 39% 訓練データ廃棄問題を解決、ImageNet FID 36.53 vs 43.84）、**crop-conditioning**（クロップ座標 $(c_{\text{top}}, c_{\text{left}})$ → 物体切れ問題解決、推論時 (0,0) で物体中心）、**multi-aspect conditioning**（40 アスペクト比 0.25-4.0、target size をフーリエ埋め込み）、(4) **改良 VAE**（ゼロから訓練、batch 256 vs 9、EMA、PSNR 24.7/SSIM 0.73/LPIPS 0.88/rFID 4.4 で SD 1.x/2.x VAE を全指標凌駕）、(5) **Refinement Model**（同じ潜在空間の別 LDM、最初の 200 ノイズ・スケールに特化、SDEdit ベース二段階生成）。**訓練**: 256² 600k steps（batch 2048）→ 512² 200k steps → multi-aspect ~1024² 面積（offset-noise 0.05）、離散時間拡散 1000 ステップ、classifier-free guidance（10% null）。**主要結果**: ユーザ研究で **SDXL w/ refinement 48.44% / SDXL base 36.93% / SD 1.5 7.91% / SD 2.1 6.71%**（refiner 付き SDXL は SD 2.1 比 7.2× 選好）。**Midjourney v5.1 を 17,153 比較で 54.9% で凌駕**、6 カテゴリ中 4 / 10 チャレンジ中 7 で凌駕または同等（PartiPrompts P2、AWS GroundTruth）。DALL-E 2 / Bing Image Creator / DeepFloyd IF / Midjourney v5.2 を質的比較で凌駕。**FID/CLIP 古典指標問題**: SDXL の FID は SD 1.5/2.1 より悪いが人間評価で明確に好まれる → Pick-a-pic（Kirstain et al., 2023）の **「COCO FID は視覚美学と負相関」** 発見を裏付け、基盤的 text-to-image モデルには新評価指標が必要。**CreativeML Open RAIL++-M License**（商用可、Apache 2.0 風）。コード + 重み完全公開。**Computer Vision wiki 初の拡散モデル系統 ingest**として、[[concepts/diffusion-model]] の主要実装例。**Stability AI の代表的成果**、後の SD3 (2024、DiT 系へ移行) / FLUX.1 (Black Forest Labs、Rombach ら独立) の前身。**弱点**: 人間の手の合成失敗、完全な写真現実性に届かず、社会的バイアス、concept bleeding（属性バインディング失敗、青い帽子 + 赤い手袋 → 入れ替わり）、長く判読可能なテキストのレンダリング困難、二段階で推論コスト増加、離散時間で offset-noise 必須、DiT 系試行は SDXL 段階では即時利益なし、古典 CLIP 系より T5 系の方が後継で優位、訓練データ非公開。
- [[sources/deepseek-ocr]] — DeepSeek-OCR（Haoran Wei, Yaofeng Sun, Yukun Li, DeepSeek-AI, 2025 Oct, arXiv:2510.18234）。**「『視覚モダリティをテキスト圧縮媒体として活用する』新パラダイムを提唱した OCR/文書理解特化型 MLLM、wiki 初の DeepSeek-AI モデル」**。**LLM 中心の視点で VLM を再定義**: 視覚を「理解対象」ではなく「テキスト情報の効率的圧縮媒体」として扱う。**2 コンポーネント**: (1) **DeepEncoder** = **[[entities/sam\|SAM-base]] (80M, patch=16, window attention で高解像度処理)** + **2 層 ConvNet (kernel=3, stride=2, padding=1, ch 256→1024, 16× ダウンサンプリング)** + **[[entities/clip\|CLIP-large]] (300M, dense global attention で意味抽出)** = 約 380M、(2) **DeepSeek-3B-MoE-A570M デコーダ**（64 routed expert + 6 活性化 + 2 shared expert = 推論時 570M 活性）。**6 つの解像度モード**: Tiny 512² → **64 トークン** / Small 640² → **100 トークン** / Base 1024² → **256 トークン** / Large 1280² → **400 トークン** / **Gundam n×640²+1024² → n×100+256 動的（n=2-9）** / Gundam-M n×1024²+1280²（継続訓練）。**Valid Token 式**でアスペクト比保持パディング考慮。**訓練データ**: OCR 1.0 70%（30M+ PDF ページ・100 言語、シーン OCR 中英各 10M、Word 3M）+ OCR 2.0（10M チャート pyecharts/matplotlib → HTML 表 + 5M 化学式 PubChem/RDKit → SMILES + 1M 平面幾何 Slow Perception）+ 一般視覚 20%（DeepSeek-VL2 風 captioning/detection/grounding）+ テキスト 10%（社内データ seq 8192）。**訓練**: Stage 1 DeepEncoder（batch 1280, lr 5e-5, cosine annealing）→ Stage 2 全訓練（HAI-LLM 20 ノード × 8× A100-40G = 160 GPU、PP=4 + DP=40、PP0 SAM+圧縮器凍結 / PP1 CLIP / PP2-3 各 6 層、batch 640, lr 3e-5, seq 8192）、**速度 70-90B トークン/日**。**主要結果**: **Fox ベンチマーク** で 10× 圧縮時 **96.5-98.5% 精度**（近似ロスレス）、20× 圧縮時 **59.1% 精度**（記憶忘却機構の根拠）。**OmniDocBench**（編集距離↓）: **100 トークン Small で 0.205（GOT-OCR2.0 の 256 トークン 0.287 を凌駕、2.56× 効率）**、Base 256 で 0.156、Large 400 で 0.117、**Gundam 795 で 0.083 SOTA（MinerU2.0 の 6790 トークン 0.133 を凌駕、8.5× 効率、dots.ocr 5545 0.125 / InternVL3-78B 6790 0.218 / Qwen2.5-VL-72B 3949 0.214 を上回る）**、Gundam-M 1853 で 0.085。**文書タイプ別**: Financial Reports 0.027 / Books 0.037 / Slides 0.08 / Newspapers 0.645（Gundam 必須）。**実用価値**: **1 台 A100-40G で 200K+ ページ/日、20 ノードで 3300 万ページ/日**（LLM/VLM 事前学習データ生成）。**質的能力**: 深層解析（チャート→HTML表 / 化学式→SMILES / 幾何→座標）、約 100 言語（アラビア語・シンハラ語等）、OCR 専門化 70% でも一般視覚理解保持。**Memory Forgetting Mechanism**（独自貢献）: 人間の記憶減衰と視覚知覚劣化の並行性を活用、古い文脈テキストを画像化して段階的縮小、最近 = 高解像度・古い = 削減解像度・非常に古い = 極度圧縮（60% 精度維持）、**LLM の長文脈問題への新解決策**として提案。**オープンソース**（GitHub）。**同著者 Haoran Wei の系譜**: Vary → GOT-OCR2.0 → DeepSeek-OCR。**Computer Vision wiki 内では Qwen 系 / InternVL 系 / Google 系（Gemma）に続く第 4 の中国系 AI ラボ系譜（DeepSeek-AI）**。**弱点**: OCR 特化で汎用 MLLM ではない（MMMU 報告なし）、SFT 段階なし、Newspapers で 0.645、20× 圧縮 60% の実用許容度不明、記憶忘却機構は概念実証のみ、OmniDocBench に評価偏重、SAM+CLIP 直列の自然画像理解汎用性不明、DeepSeek-VL2 系列との分業不明確。
- [[sources/gemma-3]] — Gemma 3（Gemma Team Google DeepMind, 2025 Mar, arXiv:2503.19786）。**「Google DeepMind の軽量オープン MLLM 第 3 世代、wiki 初の Google 系オープン MLLM」**: Gemini 2.0 と co-design された、**1B/4B/12B/27B の 4 サイズ × PT/IT の 8 公開モデル**。**1B は text-only、4B/12B/27B はマルチモーダル**（[[entities/siglip\|SigLIP]] 400M variant を 4B/12B/27B で **共有 + 凍結**、896² 固定 + **4×4 average pooling で 256 トークン固定圧縮**）。**Pan & Scan (P&S)**: LLaVA 風タイル分割（推論時のみ、非正方形・高解像度画像を等サイズクロップに分割、無効化可）、**InfoVQA +17.0 / DocVQA +4.8 の効果**。**5:1 local:global attention + sliding window 1024**: Gemma 2 の 1:1 から変更、**32K 文脈で KV キャッシュ・オーバーヘッドを 60% → <15% に削減**、long context は global 層のみ対応。**128K 文脈**（1B は 32K、global RoPE 10k→**1M**、local 10k 維持、32K→128K スケーリング係数 8）。**QK-norm**（Gemma 2 の soft-capping 置換、Chameleon / Olmo 2 着想）。**GQA + RMSNorm post/pre-norm**。**Vision encoder の埋め込み事前計算**で言語モデル訓練コスト 0。**QAT（Quantization Aware Training）**: 5,000 ステップで per-channel int4 / per-block int4 / SFP8 の 3 形式提供、**27B Int4 で 14.1 GB**（消費者 GPU 可）。**事前学習**: 27B **14T トークン** / 12B 12T / 4B 4T / 1B 2T、Gemini 2.0 と同じ SentencePiece **262k 語彙**、**全モデル知識蒸留**（256 ロジット/トークン）、Pathways + ZeRO-3 + GSPMD + MegaScale XLA。**事後学習**: 改善版 BOND + WARM + WARP RL、重み平均報酬モデル + 人間フィードバック + コード実行 + 数学 ground-truth。**主要結果**: **LMSys Chatbot Arena Elo 1338（rank 9、Gemma 3 27B IT）**で DeepSeek-V3 (671B/37B MoE, 1318)、Llama-3.1-405B (1269)、Qwen2.5-72B (1257) を **より少ないパラメータで凌駕**、Gemma 2 (1220) から **+118 Elo 飛躍**。MMLU-Pro 67.5 / **MATH 89.0**（Gemini 1.5 Pro 86.5 超え、Gemini 2.0 Pro 91.8 に肉薄）/ MMMU val **64.9** / FACTS Grounding 74.9 / HumanEval 87.8 / GSM8K 95.9。マルチモーダル（IT with P&S、27B）: MMMU **64.9** / DocVQA **86.6** / InfoVQA **70.6** / MathVista **67.6** / AI2D 84.5 / ChartQA 78.0。**PaliGemma 2 比較**: 27B が DocVQA +4.4 / InfoVQA +14.4 / TextVQA +8.1 / ChartQA +12.1 で **PaliGemma 2 27B を圧倒**、4B/12B は 10× 安価に転送。長文脈: RULER 32K **91.1** / 128K 66.0（劣化）。動画 16 frames 制約: Perception Test 58.1 / ActivityNet-QA 52.8。**Gemma3-4B-IT が Gemma2-27B-IT に匹敵、Gemma3-27B-IT が Gemini-1.5-Pro と同等**。**Gemma License（Apache 2.0 風、商用可）**。HF: `google/gemma-3-{1b,4b,12b,27b}-{pt,it}`。**[[questions/vit-dynamic-resolution-evolution]] のタイル分割路線の最新形**として LLaVA → InternVL 1.5 → Gemma 3 の系譜を完成、**Qwen の根源的解決（2D-RoPE で ViT 改造）と対照的な実装的解決**。**Memorization** は全 Gemma 3 で前世代より桁違いに低く、個人情報は記憶化出力に観察されず。**弱点**: 固定 896² + P&S の保守性、256 トークン圧縮で文書理解に情報損失、128K で RULER 27B 66.0（Qwen3-VL 1M YaRN 99.5% に大きく劣る）、MMMU で Gemini 1.5 Pro -1.0 / Gemini 2.0 Pro -7.8、動画 16 frames 制約、**MoE 採用なし**（InternVL 3.5 241B-A28B / Qwen3-VL 235B-A22B と対照的）、訓練データ非公開、PaliGemma 2 との分業不明確。
- [[sources/qwen3-5-omni]] — Qwen3.5-Omni（Qwen Team Alibaba Group, 2026, arXiv:2604.15804）。**「Qwen-Omni シリーズ最新版、テキスト + 画像 + 音声 + 動画の完全 omnimodal 統合」**: **Thinker-Talker 構造**（テキスト生成 = Thinker / 音声生成 = Talker）を継承しつつ、(1) **Thinker と Talker の双方が Hybrid Attention MoE**（Qwen3.5 ベース、Gated Delta Net 採用、KV キャッシュ I/O 削減）、(2) **256K ネイティブ文脈**（10 時間音声 / 400 秒 720P 動画 / **1 億時間音声-視覚データ事前学習**）、(3) **AuT (Audio Transformer)** をゼロから 4000 万時間音声で学習（6.25Hz トークンレート、20+ 言語、中英多言語 3.5:3.5:3、動的注意ウィンドウサイズ）、(4) **マルチコードブック RVQ コーデック**で単一フレーム即時音声合成（MTP モジュール + Code2Wav ConvNet）、(5) **ARIA (Adaptive Rate Interleave Alignment)** で MFA や固定交互配置を置換、適応的レート制約でテキスト-音声整合を動的化（単語スキップ・誤発音・数字レンダリング曖昧化を解消）、(6) 多言語 **音声入力 113 言語+方言（74 言語 + 39 中国方言）/ 音声出力 36（29 言語 + 7 中国方言）/ テキスト 201**、(7) **Audio-Visual Vibe Coding**（音声-視覚命令から直接実行可能コードを生成する出現能力）。**Plus（フラッグシップ、数千億パラメータ）+ Flash の 2 バリアント**。**3 段階事前学習**: S1 Encoder Alignment（Vision/Audio Encoder、LLM 凍結）+ S2 General（~4T トークン: テキスト 0.92T + 音声 1.99T + 画像 0.95T + 動画 0.14T + 動画-音声 0.29T、seq 32K）+ S3 Long Context（seq **262K**）。**事後学習**: Thinker 3 段階（Specialist Distillation + **On-Policy Distillation (OPD)**[テキスト→音声蒸留] + Interaction-Aligned RL）+ Talker 4 段階（General + Long-Context CPT 64K + DPO/GSPO RL + Speaker Fine-tuning）。**First-Packet Latency**: 音声入力 235ms (Flash) / 435ms (Plus)、動画入力 426ms / 651ms。**主要結果**: **215 音声・音声-視覚サブタスクで SOTA**。音声: MMAU 82.2 / MMSU 82.8 / RUL-MuchoMusic 72.4（Gemini +12.8）/ VoiceBench 93.1（+4.2）/ Fleurs ASR 6.55 WER（Gemini 7.32）/ Librispeech clean 1.11 WER（Gemini 3.36）/ **Kespeech 中国方言 3.46 WER（Gemini 23.67 = 6.8× 改善）**。視覚: 同サイズ純粋 VL の Qwen3.5-Plus-Instruct と同等、**動画 6/6 ベンチで上回る**（VideoMME 81.9 / MLVU 86.8 / MVBench 79.0 / LVBench 71.2 / MME-VideoOCR 77.0 / MMVU 67.5）、RealWorldQA **+5.0**（共同 video-audio 学習効果）、SLAKE 医療 VQA 84.7、EmbSpatialBench 85.4。音声-視覚: **DailyOmni 84.6 SOTA / Qualcomm IVD 68.5（Gemini +2.3 実世界 AV 対話）/ Omni-Cloze 64.8（+7.6 AV キャプション）**。音声生成: **SEED-TTS test-en 1.26 WER SOTA**、29 言語中 22 で多言語 WER 最低、**クロス言語 12 方向中 10 で SOTA**、zh-to-ko で CosyVoice3 から **72% 削減**。多言語 ASR: **FLEURS 60 言語平均 6.6 WER**（Gemini 7.3 / GPT-4o-Transcribe 10.4）、広東語 2.2（Gemini 6.3）。多言語翻訳: en2xx 33.8（Gemini 31.8）/ zh2xx 21.4（Gemini 19.6）、**広東語 +15.6 BLEU**。**純粋テキスト**: 同サイズ Qwen3.5-Plus-Instruct と同等（MMLU-Pro 85.9 vs 86.8）、**IFBench で純粋 LLM を上回る 52.6 vs 51.1**（OPD + Interaction-Aligned RL の正の効果）。**API のみ公開**（モデル重み非公開、cf. Qwen3-VL は Apache 2.0）。Qwen ファミリーの **「Omni 統合路線」**（vs Qwen3-VL の「VL 純化路線」）として並列発展。**弱点**: OmniGAIA ツール使用で Gemini に -11.7、VideoMME w/ audio で -5.3、Plus と Flash の 2 バリアントのみ、訓練データ非公開、4000 万時間音声学習は独自データ依存。
- [[sources/qwen3-vl]] — Qwen3-VL（Qwen Team Alibaba Group, 2025 Nov, arXiv:2511.21631）。**「Qwen-VL シリーズ第 4 世代、Qwen ファミリーが商用最先端と完全に肩を並べた決定的瞬間」**: [[entities/qwen2-5-vl\|Qwen2.5-VL]] の Window Attention + MRoPE absolute time + QwenVL HTML を発展させ、**3 つの主要構造革新**: (1) **Interleaved MRoPE**（temporal/height/width を埋め込み次元に**交互配置**して周波数スペクトルを均衡化、Qwen2-VL/2.5-VL の塊状分割を改善、長動画理解強化）、(2) **DeepStack**（Meng et al., 2024、ViT 中間 3 層の視覚特徴を LLM 最初 3 層に注入する多層融合、追加文脈長なし、アブレ +1.3 ポイント平均）、(3) **テキスト・ベース時間整合**（MRoPE absolute time を捨て、`<3.0 seconds>` のような**明示的タイムスタンプ・トークン**に置換、秒形式 + HMS 形式、Charades-STA mIoU 63.5 で Qwen2.5-VL 50.9 +12.6）。**6 サイズ × 2 バリアント = 12 公開モデル**: dense（**2B/4B/8B/32B**）+ MoE（**30B-A3B / 235B-A22B フラッグシップ**）× Instruct（non-thinking）+ Thinking（OpenAI o1 系の長 CoT 推論）。**Vision Encoder**: **SigLIP-2 から継続学習**（Qwen2.5-VL のゼロから路線放棄）、SigLIP2-SO-400M（235B/32B/30B-A3B/8B）/ SigLIP2-Large 300M（2B/4B）、CoMP 方法論の 2D-RoPE + 絶対位置補間。**256K ネイティブ文脈**（YaRN 拡張で **1M トークン = 2 時間動画**まで外挿、Needle-in-a-Haystack 256K で 100% / 1M で 99.5%）。**4 段階事前学習**: S0 Vision-Language Alignment（**Merger のみ** 67B / 8K）+ S1 Multimodal Pre-Training（All 1T / 8K）+ S2 Long-Context（All 1T / **32K**）+ S3 **Ultra-Long-Context Adaptation**（All 100B / **262K**）= 累積 ~2.2T。事前学習データ 8 カテゴリ（Image Caption + Interleaved / Knowledge / OCR 多言語 **39 言語** / Grounding [0,1000] 正規化座標復帰 / 9-DoF 3D bbox Omni3D / Code Multimodal / Video Length-Adaptive Sampling / STEM 6000 万 K-12 演習 + 1200 万 CoT / Agent GUI + Function Calling + Search）。事後学習: SFT（120 万、32K → 256K、non-thinking + thinking 分岐）+ **Strong-to-Weak Distillation**（テキスト専用、Off-policy → On-policy KL）+ RL（**SAPO アルゴリズム**、Reasoning RL + General RL）+ **Thinking with Images**（2 段階、Qwen2.5-VL-32B で 10K → 蒸留 120K、3 報酬: Answer Accuracy + Multi-Turn Reasoning + Tool-Calling）。**平方根再重み付け**でテキストとマルチモーダルのバランス改善。**主要結果**（235B-A22B）: **MathVista 85.8 / MathVision 74.6 / MathVerse_mini 85.0 SOTA**、MMMU 80.6（GPT-5 84.2 に -3.6）、**OCRBench 920 / OCRBench_v2 zh 63.5（GPT-5 37.7 +25.8 圧倒）**、OmniDocBench en **0.143 SOTA**、MMLongBench-Doc **57.0 SOTA**、**MUIRBENCH 80.1 SOTA**、CountBench 93.7、**ODinW-13 48.6**（Qwen2.5-VL 43.1 +5.5）、**VSI-Bench 62.7 / EmbSpatialBench 84.3 / RefSpatial 69.9**、**Charades-STA mIoU 64.8 SOTA**（Qwen2.5-VL +13.9）、V\* **93.7+ with tools**（GPT-5 72.8 +20.9）、**ScreenSpot Pro 62.0**（Qwen2.5-VL 43.6 +18.4）、**OSWorld 38.1**（Qwen2.5-VL 8.83 = 4.3× 飛躍、Claude Opus 4 44.4 に -6.3）、**AndroidWorld 63.7**（Qwen2.5-VL 35 +28.7）、OSWorldG 68.3、WindowsAA 32.1。**純粋テキスト**: Qwen3-235B-A22B-Instruct-2507 純粋 LLM 比で **AIME-25 +4.4 / HMMT-25 +2.0 / LiveCodeBench v6 +2.5** など、「マルチモーダル化で言語が強くなる」継続実証。**Apache 2.0**。HF: `Qwen/Qwen3-VL-{2B/4B/8B/32B/30B-A3B/235B-A22B}-{Instruct/Thinking}`。**Interleaved MRoPE + DeepStack + テキスト・タイムスタンプ + MoE + thinking モード**は MLLM 業界の新標準。**弱点**: MMMU で GPT-5 -3.6、We-Math で Gemini Thinking -5.8、Video-MME / MLVU で商用にやや劣る、OSWorld で Claude に -6.3、訓練データ非公開、6×2=12 モデル管理コスト大。
- [[sources/qwen2-5-vl]] — Qwen2.5-VL（Bai et al., Qwen Team Alibaba Group, 2025 Feb, arXiv:2502.13923）。**「Qwen-VL シリーズ第 3 世代、視覚言語モデルから視覚エージェントへの転換点」**: [[entities/qwen2-vl\|Qwen2-VL]] の Naive Dynamic Resolution + M-RoPE を基盤に、**4 つの主要技術**: (1) **Window Attention を組み込んだ ViT をゼロから学習**（32 層中 4 層のみ完全自己注意 = インデックス {7,15,23,31}、残り 28 層は 112×112 ウィンドウ、計算量を 2 次 → 線形化、DataComp + 社内データで初期化、RMSNorm + SwiGLU で LLM 系統一）、(2) **MRoPE Aligned to Absolute Time**（temporal ID をフレーム番号ではなく **絶対秒数**に揃える、異なる FPS の動画で一貫した時間整合を学習、Charades-STA mIoU 50.9 で GPT-4o 35.7 を +15.2 圧倒）、(3) **動的 FPS サンプリング + 動画 3D パッチ分割**（連続 2 フレーム grouping、秒形式 + hmsf 形式タイムスタンプ）、(4) **事前学習データ 1.2T → 4.1T トークンへ拡張**（QwenVL HTML フォーマットで文書統一表現、ABC notation 楽譜・SMILES 化学式、10,000+ カテゴリ・存在しないカテゴリ合成、PixMo 点グラウンディング、多言語 OCR 10 言語）。**3B / 7B / 72B の 3 サイズ**（全サイズで同一 ViT 共有、LLM は Qwen2.5 base）。**3 段階事前学習**（1.5T Visual + 2T Multimodal + 0.6T Long-Context、Stage3 で seq 8K → 32K）+ **2 段階事後学習**（SFT 200 万 + DPO、ViT 凍結、データフィルタリング Qwen2-VL-Instag で 8 ドメイン × 30 サブカテゴリ階層分類 + 棄却サンプリング）。**主要結果**（72B）: MMMU **70.2**（Qwen2-VL +5.7、GPT-4o 69.1 超え、初の 70 突破）/ MathVista **74.8**（GPT-4o 63.8 +11.0）/ MATH-Vision **38.1**（Qwen2-VL から +12.2、GPT-4o +7.7）/ MathVerse **57.6** / MMBench-EN **88.6 SOTA** / MMStar **70.8 SOTA** / MMVet **76.2 SOTA** / MME-RealWorld **63.2 SOTA** / DocVQA **96.4** / InfoVQA **87.3** / OCRBench **885** / OCRBench_v2 en/zh **61.5/63.7**（Gemini 1.5-Pro 比 en +9.6 / zh +20.6 圧倒）/ CC-OCR **79.8** / CharXiv DQ **87.4** / SEED-Bench-2-Plus **73.0** / CountBench **93.6 SOTA** / ODinW 43.1 / RefCOCO 92.7（InternVL2.5-78B 93.7 にやや後れ）/ MVBench **70.4** / MMBench-Video **2.02** / LVBench **47.3**（GPT-4o 30.8 +16.5）/ EgoSchema **76.2** / MLVU **74.6**（GPT-4o 64.6 +10.0）/ TempCompass **74.8** / Charades-STA mIoU **50.9 SOTA**（時間グラウンディング）。**GUI エージェント**: ScreenSpot Pro **43.6**（Qwen2-VL 1.6% の +42 ポイント、27× 飛躍、Aguvis-72B 23.6 圧倒）/ Android Control High EM **67.36** / Low EM **93.7** / AndroidWorld SR **35%**（GPT-4o 超え）/ MobileMiniWob++ **68%** / OSWorld 8.83（Claude 14.90 に後れる）。**純粋テキスト**: Qwen2.5-72B（純粋 LLM）とほぼ同等を維持（マルチモーダル化による劣化なし）、LiveBench / MultiPL-E / IFEval で上回る。**Apache 2.0**（3B/7B）/ Qwen License（72B）。HF: `Qwen/Qwen2.5-VL-{3B/7B/72B}-Instruct`。**Window Attention + MRoPE absolute time + 統一文書 HTML + GUI agent パラダイム** が MLLM 業界の新標準として広く採用、後の Qwen3-VL の基盤。**弱点**: MMMU-Pro 51.1（GPT-4o 51.9 に -0.8）、Video-MME で Gemini 1.5-Pro に -1.7/-2.2、OSWorld デスクトップで Claude に -6.07、ViT ゼロから学習は計算コスト大、訓練データ非公開（cf. InternVL 3 は公開）、依然として「LLM 事前学習 → MLLM 適応」パラダイム（cf. InternVL 3 の Native Multimodal Pre-Training）。
- [[sources/qwen2-vl]] — Qwen2-VL（Wang et al., Qwen Team Alibaba Group, 2024 Sept, arXiv:2409.12191）。**「Qwen-VL シリーズ第 2 世代、Naive Dynamic Resolution + M-RoPE で『任意解像度』を初めて本格実現」**: [[entities/qwen-vl\|Qwen-VL]] 初代の 448² 固定 + 256 トークン固定圧縮を捨て、**ViT の絶対位置埋め込みを 2D-RoPE に置換 + MLP で隣接 2×2 トークン圧縮** で任意解像度を実現。**M-RoPE**（temporal/height/width の 3 成分回転位置埋め込み）でテキスト・画像・動画の位置情報を統一モデル化、**学習 16K トークン → 推論 80K トークンまで外挿可能**。**統一画像/動画処理**（2 fps + 深さ 2 の 3D 畳み込みで 3D チューブ化、動画あたり 16K トークン上限）で **20 分以上の動画理解**を実現。**2B / 7B / 72B の 3 サイズ**（全サイズで 675M ViT 共有、DFN 初期化、LLM は Qwen2）。3 段階学習、累積 **1.4T トークン**（テキスト・トークンのみ監督）。新しい特殊トークン: `<|vision_start|>` / `<|vision_end|>` / `<|box_start|>` / `<|box_end|>` / `<|object_ref_start|>` / `<|object_ref_end|>`。**主要結果**（72B）: DocVQA **96.5** / InfoVQA **84.5** / AI2D **88.1** / TextVQA **85.5** / OCRBench **877** / MTVQA **30.9** / RealWorldQA **77.8** / MMBench-CN **86.6 SOTA** / MathVista **70.5**（GPT-4o 63.8 +6.7）/ RefCOCO val **93.2**（汎用 SOTA、専門 UNINEXT-H 92.6 を上回る、Qwen-VL 初代 89.4 から +3.8）/ MVBench **73.6** / EgoSchema **77.9**（GPT-4o 72.2 +5.7）/ Video-MME 71.2/77.8 / 多言語 OCR で韓 / 日 / 仏 / 独 / 伊 / 露 / 越で GPT-4o を上回る（アラビア語のみ -5.2）。**エージェント**: Function Calling TM 93.1 / EM 53.2、AITZ UI 操作 EM 72.1（GPT-4o 35.3 を圧倒）、Number Line / EZPoint で 100%、ALFRED SR 67.8 で専門 ThinkBot 超え。**弱点**: MMMU で GPT-4o に -4.6、MathVision -4.5、R2R ナビ -27.3（3D 空間モデリング）、依然として「LLM 事前学習 → MLLM 適応」の保守的パラダイム（cf. [[entities/internvl-3\|InternVL 3]] の Native Multimodal Pre-Training）。**Naive Dynamic Resolution + M-RoPE は MLLM 業界の事実上の標準**として広く採用、後の Qwen2.5-VL / Qwen3-VL の基盤。
- [[sources/qwen-vl]] — Qwen-VL（Bai et al., Alibaba Group, 2023 Aug, arXiv:2308.12966）。**「Qwen-VL シリーズ第 1 世代、Alibaba 製 MLLM の起点」**: Qwen-7B + OpenCLIP ViT-bigG + Position-aware VL Adapter（単層クロスアテンション + **256 学習可能クエリ** + 2D 絶対位置エンコーディング）= **9.6B**。**3 段階学習**（Stage1: 224² で LLM 凍結 + ViT/Adapter 学習、Stage2: 448² で 7 タスク同時学習、Stage3: ViT 凍結 SFT）。**特殊トークン**: `<img></img>` / `<box></box>` / `<ref></ref>`、**バウンディング・ボックスは [0, 1000) 文字列正規化**で位置語彙不要に。データ: Web クロール 50B → 14B 浄化（En 77.3% / Zh 22.7%）。**主要結果**: Flickr30K (0-shot) **85.8 CIDEr**（Flamingo-80B 67.2 を 9.6B モデルで上回る）/ VQAv2 79.5 / OKVQA 58.6 / GQA 59.3 / TextVQA 63.8 / DocVQA 65.1 / RefCOCO val **89.36**（Shikra-13B 87.83 を上回る）/ RefCOCO+ val 83.12 / RefCOCOg val 85.58 / GRIT refexp 78.22 / TouchStone En 645.2 + Cn **401.2**（中国語マルチモーダル対話の事実上の標準）/ MME Perception 1487.58 + Cognition 360.71 / SEED-Bench All 58.2。**`<box>`/`<ref>` 特殊トークン設計は MLLM 業界の事実上の標準**になり、後の Qwen2-VL / Qwen2.5-VL の基盤。InternVL シリーズ（[[sources/internvl]]）と並ぶ 2023〜2025 年オープンソース MLLM 2 大系譜の片翼。**弱点**: 解像度 448² 固定（後の Qwen2-VL で動的解像度に発展）、256 トークン固定圧縮、Stage1 で LLM 凍結（cf. InternVL 3 の Native Multimodal Pre-Training）、単一モデル規模のみ。
- [[sources/vision-transformer]] — ViT（Dosovitskiy et al., Google Brain, 2020 arXiv:2010.11929 / ICLR 2021）。**Vision Transformer 原典論文**。「画像を 16×16 のパッチに切り分け、各パッチを単語のように扱って標準的な Transformer に流し込むだけ」というシンプルなアイデアで、JFT-300M 級のデータで事前学習すれば CNN を上回ることを示した。**ImageNet 88.55% / VTAB 77.63%**、BiT-L の 1/4 以下の計算（2.5k vs 9.9k TPUv3-core-days）。3 つの普遍的スケーリング知見: **(1) データ・サイズが帰納バイアスの差を埋める**（IN では BiT > ViT、JFT では ViT > BiT）、**(2) 小データでは ViT-L < ViT-B、大データで初めて逆転**、**(3) 試した最大スケールでも飽和しない**。ViT-B/L/H の命名規約と該当パラメータ数（86M/307M/632M）を確立。学習可能 1D 位置埋め込み・[CLS] トークン・pre-norm・GELU MLP の標準構造を確立。第 1 層の埋め込みフィルタが Gabor 風基底に自然収束、位置埋め込みが 2D 構造を学習、attention 距離が CNN 受容野類比的に増加するという内部分析（§4.5）も提供。Appendix で §4.6 マスク・パッチ予測（後の MAE / iBOT / BEiT の出発点）、§D.4 位置埋め込み比較（1D ≈ 2D ≈ 相対）、§D.7-8 attention distance / map の詳細分析を含む。**本 wiki の最も基礎的な原典の 1 つ**: 登録済みのほぼすべての視覚モデル（DINO/MAE/iBOT/DINOv2/DINOv3/CLIP/SigLIP/SAM/PE/Qwen-VL 系/InternVL 系/Gemma 3/DeepSeek-OCR）が ViT を視覚バックボーンとして採用。JFT-300M 非公開・密予測の弱さ・$\mathbb{O}(N^2)$ 計算量・小データでの脆弱性は後続研究（Swin/Hiera/DeiT/MAE/RoPE）で順次解決。
- [[sources/dfn]] — DFN（Fang et al., Apple + University of Washington, 2023、arXiv:2309.17425 / ICLR 2024）。**データ・フィルタリング・ネットワーク**パラダイムの原典。**Qwen2-VL / Qwen2.5-VL の 675M ViT 初期化に DFN-CLIP が採用された MLLM の視覚基盤原典**。**鍵となるアイデア**: データセット自体を訓練対象とせず、**小さな CLIP（HQITP-350M で訓練した ViT-B/32 + OAI-Init + COCO/Flickr/IN ファインチューン）で DataComp 12.8B から上位 15% を選別し DFN-2B**（2B サンプル）**を生成**、さらに 42B プールから **DFN-5B**（5B）を生成。**主要結果**: DFN-5B で訓練した ViT-H/14 が **ImageNet ゼロショット 84.4%**（378² 解像度、224² では 83.4%）、当時 LAION-2B (73.1%) / DataComp-1B (79.2%) / OpenAI WIT (75.5%) / MetaCLIP (80.5%) / WebLI/SigLIP (83.1%) を凌駕。**3 つの非自明な発見**: (1)「フィルタ性能 ≠ ImageNet 性能」（IN -30% でもフィルタとして同等）、(2)「データ品質が決定的」（CC12M を少量汚染でフィルタ性能即座崩壊）、(3)「バイナリ分類器より CLIP フィルタ」（IN top-1 0.289 vs 次点 M3AE 0.237）。**計算効率**: DFN-2B ViT-L/14 が LAION-2B ViT-G/14 を IN +1.5% で **16× 少ない計算**で上回る、DFN-2B ViT-B/16 が OpenAI ViT-L/14 と競争的で **4× 少ない計算**。**VQA 実証**: DFN-2B ViT-L が BLIP-2 で OpenAI WIT を VQAv2 +2.8/GQA +1.3/OKVQA +2.8 で上回る（**MLLM 視覚エンコーダとしての有効性の根拠**）。**§4.2 で公開データのみでも競争的 DFN 構築可能**（CC12M+CC3M+SS15M で OpenAI WIT を超える）。**Apple Hugging Face**: apple/DFN5B-CLIP-ViT-H-14-378 / DFN5B-CLIP-ViT-H-14 / DFN2B-CLIP-ViT-L-14 / DFN2B-CLIP-ViT-B-16 を公開、DFN-2B データセットも公開。**「データセット設計はモデル設計と同じツールを使える」というデータ中心 AI ムーブメントの代表**。**Apple 系の wiki 第 1 弾**（AIMv2 などの Apple ViT 系の前任）。
- [[sources/yarn]] — YaRN（Peng et al., Nous Research / EleutherAI, 2023、arXiv:2309.00071 / ICLR 2024）。**RoPE 拡張系の決定的論文、Llama 2 を 4k → 128k に拡張**。**コミュニティ駆動の発見（Reddit / GitHub）を学術化**: PI (Chen 2023, Kaiokendev) → NTK-aware (bloc97 Reddit) → NTK-by-parts (bloc97 GitHub) → Dynamic NTK (emozilla Reddit) → **YaRN = NTK-by-parts + pre-softmax attention 温度スケーリング** の 5 世代を統合整理。**訓練効率の劇的改善**: 400 ステップ・400M トークン（元事前学習データの **0.1% 未満**）で達成、Code Llama 比 **10× 少ない**訓練データ、PI 比 **2.5× 少ない**訓練ステップ。**3 主要な技術観察**: (1) RoPE 補間の **ブラインド (PI/NTK-aware) vs ターゲット (NTK-by-parts/YaRN) の区別**、波長 $\lambda_d=2\pi b^{2d/\|D\|}$ ごとの戦略変更、(2) **「Dynamic NTK」は推論時に s を動的更新**してファインチューンなしでも文脈拡張（Qwen 7B 採用）、(3) **attention 温度 $\sqrt{1/t}=0.1\ln(s)+1$ をコード変更なしに RoPE 埋め込みスケーリングで実装**、推論コストゼロ。**主要結果**: Llama 2 7B/13B **128k で perplexity 2.37**（Code Llama 2.71、Together PI 爆発）、**Passkey 検索 128k で 99.4%**、ARC-c/HellaSwag/MMLU で Code Llama (NTK) 比 大幅優位（Code Llama は MMLU 43.8→31.1 劣化）、**Mistral 7B も同様に 128k 拡張**、Dynamic-YaRN は ファインチューンなしでも Dynamic-PI 凌駕。**Vision/MLLM での意義**: [[entities/qwen3-vl|Qwen3-VL]] の **256K ネイティブ + 1M YaRN 外挿**（Needle 256K 100% / 1M 99.5%）の理論的基盤、Qwen2-VL の 16K→80K 外挿、SAM 2 / DINOv3 / M-RoPE 系の長系列拡張の参考に。**著者 Bowen Peng は Reddit /u/bloc97**（NTK-aware の最初の発見者）で、Nous Research + EleutherAI コミュニティから生まれた論文。本論文の真の貢献は「新規アイデアの提案」より「**コミュニティ発見の学術化と統合**」。
- [[sources/roformer]] — RoFormer（Su et al., Zhuiyi Technology, 2021、arXiv:2104.09864 / Neurocomputing 2024）。**RoPE（Rotary Position Embedding, 回転位置埋め込み）の原典論文**。Transformer の self-attention に「位置情報を回転行列の乗算として注入する」新しい位置エンコーディング手法を提案。query/key ベクトルに角度 $m\theta$ の回転を適用すると内積に相対位置 $m-n$ が自然に現れる。3 つの核心性質: **乗算的（線形 attention と両立）/ 長期減衰（$\theta_i = 10000^{-2i/d}$ で保証）/ パラメータ追加ゼロ**。実験 4 種: WMT14 英独翻訳（BLEU 27.3 → 27.5）/ BERT 事前学習（MLM 損失収束高速化）/ GLUE 6 タスク（3/6 で BERT 超え、SST-2/QNLI/MNLI で負け）/ Performer 線形 attention（収束加速）/ 中国語長文 CAIL2019-SCM（1024 で WoBERT +1.5%）。**現代 LLM（LLaMA/Mistral/PaLM/Qwen/DeepSeek）と Vision/MLLM（DINOv3/SAM 2/Qwen-VL 系）の事実上の標準**となった重要論文。論文自体は NLP 向けだが、本 wiki にとっては **任意解像度・任意系列長対応の理論的基礎**として最重要原典の 1 つ。Huggingface 統合済み。**著者 Jianlin Su は中国 Zhuiyi Technology の研究者で、ブログ記事「Transformer 升级之路」で先に発想を公開していたため、英語圏への浸透が遅れた**経緯あり。
- [[sources/foundational-models-vision-survey]] — Foundational Models in Vision Survey（Awais et al., MBZUAI ら, arXiv:2307.13721 / IEEE TPAMI）。**CV 基盤モデルを「テキスト・プロンプト型 / 視覚プロンプト型 / 異種モダリティ型 / 身体性型」の 4 軸で体系的に分類**、CLIP / SAM / PaLM-E など 100+ モデルを共通の分類体系（taxonomy）に位置付けた survey。テキスト・プロンプト型はさらに CL / Generative / Hybrid / Conversational の 4 サブカテゴリに、汎用 vs 視覚グラウンディングで 2 分割。Sec. 2 で 4 種アーキテクチャ・スタイル（Dual-Encoder / Fusion / Encoder-Decoder / Adapter LLM）を定義。本 wiki の VL 基盤モデルすべて（CLIP / SigLIP / GLIP / Grounding-DINO / Qwen-VL 系 / InternVL 系 / Gemma 3 / DeepSeek-OCR / SAM 系等）に「鳥瞰図」を与える。**スコープ外**: 純粋画像 SSL（DINOv2/DINOv3/MAE 等）と画像生成系（拡散モデル等）は対象外（明示的に除外）。
- [[sources/internvl-3-5]] — InternVL 3.5（Wang et al., InternVL Team Shanghai AI Lab, 2025 Aug, technical report）。**「InternVL シリーズ第 8 世代、商用 GPT-5 との差をオープンソース最小に縮める」**: GPT-5（2025 Aug 7）直後の発表で **GPT-5 との Aggregate 差 3.9%** という当時オープンソース最小。**3 つの中核イノベーション**: (1) **Cascade Reinforcement Learning**（offline RL = [[entities/mpo\|MPO]] で warm-up + online RL = **GSPO（Geometric mean Sequence-level PPO、reference model 制約なし、トークン単位の geometric mean importance ratio）** で精緻化、coarse-to-fine 戦略、**GSPO 単独の半分の GPU 時間で +2.1 ポイント上回る**）、(2) **Visual Resolution Router (ViR) + Visual Consistency Learning (ViCO)**（patch ごとに 1/4 or 1/16 圧縮率を semantic richness で動的選択、視覚トークン 50% 削減で性能 99% 維持）、(3) **Decoupled Vision-Language Deployment (DvD)**（ViT/MLP/ViR を vision server、LLM を language server に分離、BF16 視覚特徴を TCP/RDMA で転送、**非同期 3 段階パイプライン** で 896 解像度時 **4.05× 推論加速**）。**MoE スケーリング初導入**: **20B-A4B（GPT-OSS-20B、OpenAI 公開モデル統合）/ 30B-A3B（Qwen3）/ 241B-A28B（最大）**。**9 サイズ × dense + MoE × Flash 効率版**で合計 18+ モデルのスイート。**LLM が Qwen3 系に統一**（InternVL 3 の Qwen2.5 から更新）。**主要結果**: MMMU **77.7（オープンソース新 SOTA、Claude-3.7-Sonnet 75.0 / Gemini-2.5-Pro 74.7 / GPT-5-nano 72.6 全部超え、GPT-5 84.2 に -6.5）**、MathVista **82.7（GPT-5 81.9 +0.8、Claude-3.7-Sonnet 66.8 / Gemini-2.5-Pro 80.9 超え）**、**VSI-Bench 69.5（GPT-5 37.5 を +32 圧倒、空間推論の新水準）**、WildVision **82.8（GPT-4o 80.6 +2.2）**、WindowsAgentArena でも GPT-4o 3.5 を圧倒、WebArena-Lite-v2 **11.7（GPT-4o 1.9 の 6×）**。**Cascade RL が全モデルサイズで SFT 単独から +6-12 ポイント改善**（2B モデルで +12.2 最大、241B-A28B で +6.5）。**Parallel Thinking（Best-of-N + VisualPRM-v1.1）でさらに +1-9 ポイント**。**マルチモーダル化で言語強化を [[entities/internvl-3\|InternVL 3]] から継続実証**（Qwen3-base 比で 1B が +6.7、241B が +2.3、16/16 ベンチで Qwen3 超え）。**全モデルとコード公開**。**弱点**: Reasoning Overall で GPT-5 に -7.2、Text Overall で -6.0、241B 全 load にエンタープライズ級 GPU 必要。**InternVL シリーズ現時点最新版**。

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
- [[translations/internvl]] — InternVL 原論文の本文和訳。Abstract + §1-5（Acknowledgement 含む、References と Appendix A.1-A.5 は除外、ユーザー指示）。図 1-4 を `<figure>` で埋め込み、表 1-12 を含む（表 7 のフル数値は概略のみ）。InternViT-6B のハイパーパラメータ探索、QLLaMA の cross-attn 設計、3 段階訓練（contrastive + ITC/ITM/ITG + SFT）、InternVL-C/G/Chat 4 つの推論モードを含む。
- [[translations/internvl-1-5]] — InternVL 1.5 原論文の本文和訳。Abstract + §1-5（References は除外）。図 1-5 を `<figure>` で埋め込み、表 1-3 を含む（表 2 は 18 ベンチ × 全モデル、表 3 は ConvBench/MMT-Bench）。動的高解像度（35 種アスペクト比 × 1-40 タイル）、Pixel Shuffle、継続事前学習（InternViT V1.0 → V1.2 → V1.5）、QLLaMA 廃止 → MLP 統一、バイリンガルデータパイプライン、2 段階訓練（pre-training + fine-tuning）を含む。GPT-4V/Gemini/Claude-3/Qwen-VL-Max との横並び 18 ベンチ比較を含む。
- [[translations/mini-internvl]] — Mini-InternVL 原論文の本文和訳。Abstract + §1-5（References は除外）。図 1-5 を `<figure>` で埋め込み（fig5 は欠番、fig6 を fig5 として配置）、表 1-13 を含む。InternViT-300M の知識蒸留（negative cosine similarity 損失、最後の K 層、CLIP-ViT-L 初期化）、Mini-InternVL の 2 段階訓練、統一ドメイン適応フレームワーク（5 種タスクの VQA 統一）、3 ドメイン（自律走行 / 医療 / リモセン）への転移結果、4 種類のアブレーション（KD 有無 / データ比率 / サンプル数 / 適応手法）を含む。
- [[translations/internvl-2-5]] — InternVL 2.5 原論文の本文和訳。Abstract + §1-8（Acknowledgement 含む、References は除外）。図 1-10 を `<figure>` で埋め込み、表 1-15 を含む（OpenCompass 順位、InternViT 全バージョン比較、Mini-InternVL を含むモデルカタログ、訓練ハイパラ、Pre-train/Fine-tune データ、Multimodal Reasoning/OCR/Multi-Image/Real-World/Comprehensive/Hallucination/Grounding/Multilingual/Video の全 9 ベンチマーク群、LLM 17 ベンチ、ImageNet 派生 6、ADE20K + COCO-Stuff）。Progressive Scaling Strategy 公式化、Test-Time Scaling、データフィルタリングパイプライン、InternViT 中間層特徴発見を含む。
- [[translations/internvl-3]] — InternVL 3 原論文の本文和訳。Abstract + §1-4 + Conclusion（References は除外）。図 1-3 を `<figure>` で埋め込み、表 1-13 を含む（7 サイズモデルカタログ、Multimodal Reasoning + Math（VisualPRM-Bo8 結果含む）、OCR/Chart/Document、Multi-Image + Real-World、Comprehensive + Hallucination、Grounding、Multilingual、Video、GUI Grounding、VSI-Bench 空間推論、LLM 17 ベンチ、V2PE δ アブレーション、MPO アブレーション）。Native Multimodal Pre-Training、V2PE 数式、MPO 損失（DPO + BCO + LM）、VisualPRM、InternEVO インフラを含む。
- [[translations/mpo]] — MPO 原論文の本文和訳。Abstract + §1-6 + Implementation Details + More Ablation Studies 概要（References と Appendix §9 のデータ例図は除外）。図 1-3 を `<figure>` で埋め込み、表 1-9 を含む（DropoutNTP データ例、PO アルゴリズム比較、SFT vs MPO、RLAIF-V vs DropoutNTP、データソース、テキスト専用ベンチ、X+ 拡張、DPO 変種比較、Dropout Ratio アブレーション、データスケール、ハイパラ）。MMPR データセット、DropoutNTP の数式、MPO の 3 損失（DPO + BCO + SFT loss）、3 種のマルチモーダル CoT を含む。
- [[translations/sdxl]] — SDXL 論文の本文 + 付録 B-J 和訳（Appendix A 謝辞と References 除外、ユーザー指示で Appendix 含む）。Abstract + §1-3 + Appendix B（Limitations）+ C（Diffusion Models 理論的背景、Probability Flow ODE / SDE / DSM / CFG）+ D（State of the Art 比較）+ E（Midjourney v5.1 比較、17153 投票 + PartiPrompts P2）+ F（FID/CLIP の信頼性問題、Pick-a-pic 裏付け）+ G（Single vs Two-Stage パイプライン）+ H（SD 1.5 vs 2.1 vs SDXL 比較）+ I（Multi-Aspect 40 アスペクト比表）+ J（チャンネル次元連結の Python コード）を含む。図 1-6 + 12 を `<figure>` で埋め込み、表 1-3 + 40 アスペクト比表を含む。Algorithm 1（size + crop conditioning パイプライン）も含む。
- [[translations/deepseek-ocr]] — DeepSeek-OCR 論文の本文翻訳。**Web Clipper による markdown 抽出が不完全（Abstract + References のみ）だったため、本文は ar5iv（https://ar5iv.labs.arxiv.org/html/2510.18234）から取得した完全版を翻訳**。Abstract + §1-10（Introduction, DeepEncoder Architecture, Decoder, Training Data, Training Pipeline, Experimental Results, Practical Value, Qualitative Capabilities, Memory Forgetting Mechanism, Conclusion）を含む。図 2-4 を `<figure>` で埋め込み（既存 VLM 比較、DeepEncoder アーキテクチャ、5 解像度モード文書例）、Fox ベンチマーク圧縮表、OmniDocBench 結果表、文書タイプ別性能表、SAM + 16× ConvNet + CLIP 構成詳細、DeepSeek-3B-MoE-A570M 仕様、Memory Forgetting Mechanism 概念を含む。
- [[translations/gemma-3]] — Gemma 3 Technical Report の本文 + 付録和訳。Abstract + §1-8 + Appendix 評価セクション（References と Contributors 名前リスト除外、ユーザー指示で Appendix 含む）。図 1（zurich-receipt 視覚相互作用例）、図 2（能力レーダー）、図 5（KV キャッシュ vs 構成）、図 6（KV キャッシュ vs 文脈長）、図 7（長文脈性能）を `<figure>` で埋め込み、表 1-18 + 評価詳細を含む（パラメータ数、訓練インフラ、QAT メモリ、Chatbot Arena Elo、Gemini/Gemma 2 標準ベンチ比較、解像度 / P&S アブレ、Appendix の事実性 / STEM / マルチモーダル / PaliGemma 2 比較 / 多言語 / 長文脈 / IT / 動画）。SigLIP 400M + Pan & Scan、5:1 local:global attention、QK-norm、QAT、知識蒸留 256 ロジットを含む。
- [[translations/qwen3-5-omni]] — Qwen3.5-Omni Technical Report の本文 + 付録和訳。Abstract + §1-8（References 除外）。図 1-3 を `<figure>` で埋め込み、表 1-15 を含む（first-packet 遅延、サポート言語/方言、テキスト/音声/視覚/音声-視覚→テキスト性能、ゼロショット/多言語/クロス言語/カスタム音声生成、多言語 ASR/翻訳全 60 言語）。Thinker-Talker 構造、Hybrid MoE、AuT、ARIA、Audio-Visual Vibe Coding、3 段階事前学習 + Thinker 3 段階 + Talker 4 段階の事後学習を含む。
- [[translations/qwen3-vl]] — Qwen3-VL Technical Report の本文和訳。Abstract + §1-7（References p.26+ は除外、Appendix なし、ユーザー指示）。図 1-3 を `<figure>` で埋め込み、表 1-12 を含む（4 段階事前学習、フラッグシップ 235B-A22B vs 商用、中規模 32B/30B-A3B vs Gemini 2.5 Flash/GPT-5-mini、小型 2B/4B/8B vs GPT-5 Nano、純粋テキスト、Vision Encoder と DeepStack のアブレ）。Interleaved MRoPE / DeepStack / テキスト・ベース時間整合の 3 構造革新、6 サイズ × 2 バリアント = 12 モデル、SAPO RL アルゴリズム、Thinking with Images（2 段階）、QwenVL-HTML + QwenVL-Markdown 文書解析、9-DoF 3D bbox Omni3D、Needle-in-a-Haystack 256K 100% / 1M 99.5% を含む。
- [[translations/qwen2-5-vl]] — Qwen2.5-VL Technical Report の本文和訳。Abstract + §1-5（References pp.16-23 は除外、Appendix なし、ユーザー指示）。図 1 を `<figure>` で埋め込み、表 1-9 を含む（モデル詳細、3 段階事前学習データ、SoTA 比較全 20 ベンチ、純粋テキスト、文書/OCR、グラウンディング、カウント、動画、GUI エージェント）。Window Attention + MRoPE absolute time + 動的 FPS + 3D パッチ分割、3 段階事前学習 + 2 段階事後学習、QwenVL HTML フォーマット（ABC notation 楽譜・SMILES 化学式）、データフィルタリング・パイプライン（Qwen2-VL-Instag 8 ドメイン × 30 サブカテゴリ + 棄却サンプリング）を含む。
- [[translations/qwen2-vl]] — Qwen2-VL 原論文の本文和訳。Abstract + §1-4 + Acknowledgements（References と Appendix A は除外、ユーザー指示）。図 1-6 を `<figure>` で埋め込み、表 1-8 を含む（モデル詳細、性能比較全 22 ベンチ、多言語 OCR、動画ベンチ、エージェント、参照表現理解、動的解像度アブレーション、M-RoPE アブレーション）。Naive Dynamic Resolution（2D-RoPE + MLP 2×2 圧縮）、M-RoPE（temporal/height/width 3 成分）、統一画像/動画処理（2 fps + 3D 畳み込み深さ 2）、3 段階学習、特殊トークン定義、システム実装詳細（PAI-Lingjun + CPFS/OSS + 3D 並列）を含む。
- [[translations/qwen-vl]] — Qwen-VL 原論文の本文和訳。Abstract + §1-6（Acknowledgement なし、References と Appendix A-E は除外、ユーザー指示）。図 1-4 を `<figure>` で埋め込み、表 1-7 を含む（モデル詳細、事前学習データ、マルチタスクデータ、画像キャプション + 一般 VQA、テキスト指向 VQA、参照表現理解、命令追従ベンチマーク）。Qwen-7B + OpenCLIP ViT-bigG + Position-aware VL Adapter (256 queries cross-attention + 2D abs pos enc)、3 段階学習、`<img>`/`<box>`/`<ref>` 特殊トークン、[0,1000) 座標正規化、ChatML 対話フォーマットを含む。
- [[translations/vision-transformer]] — ViT 原論文の本文和訳。Abstract + §1-5（References と Appendix A-D は除外、ユーザー指示「ingestして」のデフォルトに従う）。図 1（モデル概要）/ 図 2（VTAB タスク内訳）/ 図 3（ImageNet 転移、データセット・サイズ依存）/ 図 5（事前学習計算 vs 性能）/ 図 6（attention 例）/ 図 7（内部表現: PCA フィルタ・位置埋め込み類似度・attention 距離）を `<figure>` で埋め込み、表 1（ViT-B/L/H 仕様）と表 2（SOTA 比較）を含む。**§3 で ViT アーキテクチャの全体（パッチ分割・[CLS] トークン・1D 位置埋め込み・Transformer エンコーダ式 1-4）**、**§4.2 で SOTA 比較（ImageNet 88.55% / VTAB 77.63%、計算 1/4）**、**§4.3 でデータ要求性（IN < I21k < JFT で順位逆転）**、**§4.4 でスケーリング研究（ViT > ResNet で 2-4× 計算効率、Hybrid との比較）**、**§4.5 で内部検査（位置埋め込みの 2D 構造創発、attention 距離が CNN 受容野類比）**、**§4.6 でマスク・パッチ予測の予備実験（MAE の出発点）**を含む。**Hybrid アーキテクチャ（CNN 特徴マップ → ViT）、Fine-tuning と高解像度（位置埋め込みの 2D 補間）も含む**。
- [[translations/dfn]] — DFN 原論文の本文 + Appendix A-G 和訳（ユーザー指示で Appendix 含む、References のみ除外）。図 1（計算スケーリング）/ 図 2（パイプライン概観）/ 図 3（フィルタ性能 vs IN）/ 図 4（データ品質効果）/ 図 5（頑健性）/ 図 6（図 3 の平均精度版）/ 図 7（図 4 の平均精度版）/ 図 8（log-scale）を `<figure>` で埋め込み、表 1-10 を含む。**§3.1 DFN 定義（filter dataset / induced dataset / induced model）、§3.3 の 3 つの鍵発見、§4 の 3 段階レシピ（HQITP-350M 訓練 + OAI-Init + COCO/Flickr/IN FT）、§4.1 BLIP-2 VQA 実証、§4.2 公開データのみで競争的 DFN、Appendix A-G の訓練ハイパラ / ImageNet 頑健性 / 公開チェックポイント / 平均精度版図 / log-scale プロット / HQITP-135M のみで構築可能な追加表**を含む。
- [[translations/yarn]] — YaRN 原論文の本文 + Appendix 和訳。Abstract + §1-6（Conclusion + Reproducibility）+ **Appendix A.1（NTK-aware 補間の導出）+ A.2（pre-softmax スケーリング perplexity 影響）+ B.1（GovReport 評価）+ B.2（Passkey 検索詳細）+ B.3（ファインチューンなしの Dynamic Scaling）+ B.4（Mistral 拡張）**（References のみ除外、ユーザー指示で Appendix 含む）。図 1（Llama スライディング・ウィンドウ perplexity）/ 図 2-4（pre-softmax 温度 t の perplexity 影響、$s=8$ 固定での全体・セグメント・argmin）/ 図 5（Dynamic-RoPE vs Dynamic-PI vs Dynamic-YaRN）/ 図 6（Mistral perplexity）を `<figure>` で埋め込み、表 1-6 を含む。**PI / NTK-aware / NTK-by-parts / Dynamic NTK / YaRN の 5 手法の系譜と数式定義、波長 $\lambda_d$ と ramp 関数 $\gamma(r)$、$\alpha=1/\beta=32$ ハイパラ、attention 温度 $\sqrt{1/t}=0.1\ln(s)+1$ の経験式と「普遍性」議論、Llama 2 7B/13B の 4k→128k 拡張、Code Llama / Together / Mistral の比較、Dynamic-YaRN のファインチューンなし有効性**を含む。
- [[translations/roformer]] — RoFormer 原論文の本文和訳。Abstract + §1-5（References 除外、Appendix なし）。図 1（RoPE 実装図解）/ 図 2（長期減衰グラフ）/ 図 3（BERT vs RoFormer / Performer w/wo RoPE 訓練損失）を `<figure>` で埋め込み、表 1-5 を含む。**§2 で既存の絶対位置・正弦波・相対位置エンコーディングの 4 系統を体系整理**、**§3.2 で 2D ケースの複素数導出と d 次元への一般化（ブロック対角の回転行列）**、**§3.3 で長期減衰性と線形 attention との両立**、**§3.4 で理論的説明（2D 導出 / 効率的実装 / 長期減衰のアーベル変換による証明）**、**§4 で 4 種の実験（WMT 翻訳 / BERT 事前学習 / GLUE / Performer / 中国語長文 CAIL2019-SCM）**、**§4.5.5 で著者自身の限界**（収束高速化の理論的説明欠如・長文優位の説明欠如・事前学習コスト）を含む。
- [[translations/foundational-models-vision-survey]] — Foundational Models in Vision Survey の本文和訳。Abstract + §1-9（References §後を除外、Appendix なし）。図 1（進化タイムライン）/ 図 2（4 アーキテクチャ・スタイル）/ 図 3（taxonomy 全体図）/ 図 4（CLIP 変種）/ 図 5（CLIP の局所化失敗例）/ 図 13（SAM）/ 図 14（SAM 医療適応）/ 図 15（PaLM-E）を `<figure>` で埋め込み、表 II（テキスト・プロンプト型モデル一覧）を含む。CL（CLIP, ALIGN, FILIP, FLIP, EVA-CLIP, OpenCLIP 等）/ Visual Grounding CL（GLIP, Grounding-DINO, RegionCLIP, OWL-ViT, OpenSeg, GroupViT）/ Generative（Frozen, Flamingo, KOSMOS, SimVLM, PaLI）/ Hybrid（FLAVA, BLIP, BLIP-2, InstructBLIP, CoCa, UNITER）/ Conversational VLM（GPT-4, MiniGPT-4, LLaVA, Video-ChatGPT, LLaMA-Adapter V2）/ Visually Prompted（SAM, SegGPT, SEEM, CLIPSeg, MedSAM, AutoSAM, MobileSAM 等）/ Generalist（Painter, VisionLLM, Prismer）/ Heterogeneous（CLIP2Video, AudioCLIP, ImageBind, MACAW-LLM, COSA, Valley）/ Embodied（PaLM-E, ViMA, MineDojo, VOYAGER, LM-Nav）/ Open Challenges 8 項（Multimodal Open-source, Evaluation, Hallucination, Multimodal Alignment, Data/Compute, FM Adaptation, Adversarial, Bias, Interpretability, Contextual Understanding, Real-world Understanding）を網羅。
- [[translations/internvl-3-5]] — InternVL 3.5 原論文の本文和訳。Abstract + §1-4（References は除外）。図 1-5 を `<figure>` で埋め込み、表 1-18 を含む（9 サイズモデルカタログ、全 35 ベンチマーク比較、Reasoning + Math、OCR/Chart/Document、Multi-Image + Real-World、Comprehensive + Hallucination、Grounding、Multilingual、Video、GUI、Embodied、SVG、LLM 16 ベンチ、Cascade RL アブレーション、Cascade RL vs MPO vs GSPO 効率、ViR Flash 性能維持、DvD + ViR 加速）。Cascade RL（MPO + GSPO）、ViR + ViCO の数式、DvD、MoE モデル設計を含む。

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
- [[entities/dfn]] — Apple の **DFN（Data Filtering Networks、2023、ICLR 2024）**。**Qwen2-VL / Qwen2.5-VL の 675M ViT 初期化に DFN-CLIP が採用された、MLLM 視覚エンコーダの直接原典**。**手法 = DFN（小さな CLIP モデルでデータ選別）+ 誘導データセット = DFN-2B（2B サンプル、DataComp 12.8B から 15%）/ DFN-5B（5B、42B プールから）**。**最高モデル DFN5B-CLIP-ViT-H-14-378 が ImageNet ZS 84.4%**（当時の SOTA、LAION-2B/DataComp-1B/WIT/MetaCLIP/WebLI 凌駕）。**HQITP-350M（Apple 内部の 357M 人間検証済みキャプション、非公開）で DFN を訓練、OAI-Init + COCO/Flickr/IN ファインチューン + 重みアンサンブル**。**3 つの非自明発見**: (1) フィルタ性能 ≠ IN 性能、(2) データ品質が決定的（汚染で即座崩壊）、(3) CLIP フィルタ > バイナリ分類器。**Apple HF**: apple/DFN5B-CLIP-ViT-H-14-{224,378} / apple/DFN2B-CLIP-ViT-L-14 / apple/DFN2B-CLIP-ViT-B-16、DFN-2B データセット公開。**[[entities/qwen2-vl|Qwen2-VL]] / [[entities/qwen2-5-vl|Qwen2.5-VL]] で採用**、[[entities/qwen3-vl|Qwen3-VL]] では SigLIP-2 に切り替え。データ中心 AI ムーブメントの代表。
- [[entities/perception-encoder]] — Meta の Perception Encoder（NeurIPS 2025）。**PEcore / PElang / PEspatial の 3 バリアント**。対比学習をスケールするだけで中間層に多目的特徴が育つことを発見、alignment tuning（[[concepts/alignment-tuning]]）で末端に引き出す。MLLM・検出・dense 予測すべてで SOTA。
- [[entities/detr]] — Facebook AI の DETR（ECCV 2020）。Transformer ベースの end-to-end 物体検出。ResNet-50/101 backbone + 6 層 encoder + 6 層 decoder + 100 個の object queries。Hungarian 二部マッチング損失で NMS 不要。DETR-DC5-R101 で COCO 44.9 AP、大物体 62.3。DETR ファミリー（Deformable DETR, DAB-DETR, DN-DETR, DINO-detector, DETA, CoDETR, MDETR）の祖。
- [[entities/dino-detector]] — IDEA/HKUST/清華大の DINO 検出器（ICLR 2023）。**DETR ファミリーの集大成**。ResNet-50/SwinL backbone + 6/6 層 + 900 queries + 4D anchor box queries + deformable attention。**CDN + Mixed Query Selection + Look Forward Twice** で 12 epoch 49.4 AP / 24 epoch 51.3 AP、**DINO-SwinL（218M params）で COCO test-dev 63.3 AP** — 初の end-to-end Transformer SOTA。SwinV2-G の 1/15 パラメータで上回る。**SSL の DINO（[[entities/dino]]）とは完全に別物**。
- [[entities/glip]] — Microsoft の GLIP（CVPR 2022）。Swin-T/L backbone + DyHead 検出ヘッド + BERT テキストエンコーダ + X-MHA deep fusion。**物体検出と phrase grounding の統一定式化** で region-word alignment を学習。5 つのバリアント（GLIP-T (A/B/C), GLIP-T, GLIP-L）。GLIP-L は 27M grounding データ（FourODs + GoldG + Cap24M）で COCO ゼロショット 49.8 AP / fine-tune 61.5 AP（SOTA）。**Open-vocab 検出パラダイムの祖**、Grounding DINO / SAM 3 / OWL-ViT の源流。
- [[entities/grounding-dino]] — IDEA + Microsoft の Grounding DINO（ECCV 2024）。Swin-T/L backbone + BERT + 6 層 feature enhancer + 6 層 cross-modality decoder + 900 queries。**[[entities/glip|GLIP]] × [[entities/dino-detector|DINO 検出器]] の正統な統合**（両グループの直接共同）。2 バリアント（Grounding-DINO-T 172M / -L 341M）。Grounding-DINO-L で COCO ZS 52.5 AP / FT 63.0 AP、ODinW ZS 26.1 SOTA、RefCOCO val 90.56 SOTA。**SAM 3 と MM-Grounding-DINO の直接の祖**。
- [[entities/yolo-world]] — Tencent + 華中科技大学の YOLO-World（CVPR 2024）。YOLOv8 backbone + CLIP-base text encoder + RepVL-PAN ネック。3 バリアント（YOLO-World-S 13M / -M 29M / -L 48M、re-parameterize 版）。LVIS ZS 35.4 AP at 52 FPS V100。GLIP-T (0.12 FPS) の 433× / Grounding-DINO-T (1.5 FPS) の 35× / DetCLIP-T (2.3 FPS) の 22× 速度。**エッジデバイスでの open-vocab 検出を可能にした初の本格的研究**。GLIP-L を疑似ラベル生成 teacher として活用。
- [[entities/grounding-dino-1-5]] — IDEA Research の Grounding DINO 1.5（2024 May）。**Pro と Edge の双子モデル**。Pro: ViT-L backbone + Grounding-20M で **COCO ZS 54.3 / LVIS-mv 55.7 / ODinW35 30.2 / Fine-tune 68.1 (LVIS-mv) / 70.6 (ODinW35) SOTA**。Edge: EfficientViT-L1 backbone + Efficient Feature Enhancer（P5 のみ cross-modal 融合）で **Orin NX 10.7 FPS / A100 TRT 75.2 FPS、LVIS-mv 36.2 AP**（YOLO-Worldv2-L 超え）。**精度志向（Pro）と実用志向（Edge）を 1 モデルスイートで統合**した IDEA Research の戦略的後継。
- [[entities/dino-x]] — IDEA Research の DINO-X（2024 Nov、Grounding DINO 1.5 の正統な後継）。**unified object-centric vision model**。Pro: ViT-L + CLIP text encoder + 3 プロンプト (Text/Visual/Customized) + 4 ヘッド (Box/Mask/Keypoint/Language) + **Grounding-100M**（GD 1.5 の 5×）。**LVIS-mv 59.8 / rare APr 63.3 SOTA** / Visual Genome CIDEr **201.8** / FSC147 カウント MAE 5.6 / Hand pose HInt SOTA。Edge: EfficientViT-L2 + Knowledge Distillation (Pro→Edge) + FP16 で **Orin NX 20.1 FPS、LVIS-mv 48.3 AP**（YOLO-Worldv2-L 33.0 を +15.3 AP 凌駕）。**Universal Object Prompt で prompt-free 検出**。**「Grounding DINO 系統の到達点」**かつ SAM 3 と並行進化する unified perception model。
- [[entities/internvl]] — OpenGVLab Shanghai AI Lab の **InternVL（CVPR 2024、InternVL 1.0）**。InternViT-6B（5.9B vanilla ViT, w=3200/d=48/MLP=12800/h=25）+ QLLaMA（8B = 多言語 LLaMA-7B + 1B 新規 96 query/cross-attn 層）。4 つの推論モード（**InternVL-C** 対比 / **InternVL-G** 対比+生成 / **InternVL-Chat (MLP)** LLaVA 互換 / **InternVL-Chat (QLLaMA)** フル構成）。Stage 1（4.98B 対）+ Stage 2（1.03B + ITC/ITM/ITG）+ Stage 3（4M SFT）の 3 段階訓練。**ViT-22B（21.7B）を 1/3.7 の 5.9B パラメータで ADE20K linear probe +12.6 mIoU 上回る**画期的成果。Vicuna-7B/13B / InternLM 接続可能。InternVL 1.2 / 1.5 / 2.0 / 2.5 / 3 シリーズの起点。
- [[entities/internvl-1-5]] — OpenGVLab Shanghai AI Lab の **InternVL 1.5（2024 April、InternVL シリーズ第 4 世代）**。InternViT-6B-448px-V1.5（5.9B、**45 層 = 元 48 層 - 3**、dynamic 448）+ **MLP プロジェクタ（~150M、QLLaMA を完全廃止）** + InternLM2-20B-Chat = **26B**。**動的高解像度**: 35 通りのアスペクト比 × 1-12 タイル訓練 / 40 タイル zero-shot テスト（4K 相当）+ Pixel Shuffle で visual token 1/4 圧縮。**継続事前学習**: V1.0（CVPR 2024, 224、48 層）→ V1.2（448 固定、45 層、Yi-34B）→ V1.5（448 dynamic、InternLM2-20B-Chat）。**18 ベンチ中 8 SoTA**（ChartQA 83.8 / OCRBench 724 / MMBench-CN 82.0 / **CCBench 69.8 で GPT-4V +23.3** / MathVista 53.5 で GPT-4V +3.6 / HallusionBench 49.3）。**ConvBench マルチターン対話では GPT-4V に大敗**（17.65 vs 39.51）。バイリンガル（英中）OCR データを PaddleOCR で擬似ラベリング。MIT + InternLM2 ライセンス。HF: `OpenGVLab/InternVL-Chat-V1-5`。
- [[entities/internvl-2]] — OpenGVLab Shanghai AI Lab の **InternVL 2.0（InternVL2、2024 Jul、InternVL シリーズ第 5 世代）**。**1B/2B/4B/8B/26B/40B/76B の 7 サイズ・スイートを初確立**（フラッグシップ **InternVL2-Llama3-76B**）。アーキテクチャは [[entities/internvl-1-5\|InternVL 1.5]] と本質的に同じ ViT-MLP-LLM（InternViT-6B-448px or InternViT-300M-448px + MLP + Pixel Shuffle + 動的解像度 1-40 タイル）、LLM は InternLM2 / Qwen2 / Llama3 を混用。**主要結果**: MMMU **62.7** / MMB-EN 86.5 / MathVista 67.0 / RefCOCO 92.2 / DocVQA 94.1。**公式 Technical Report リリースなし**で source ページは未取り込み、後続 [[entities/internvl-2-5]] / [[entities/mini-internvl]] / [[entities/internvl-3]] でのベースライン引用が主要情報源。シリーズ位置: 1.5 → **2.0** → 2.5（MMMU 70% 突破）。
- [[entities/mini-internvl]] — OpenGVLab Shanghai AI Lab の **Mini-InternVL（2024 Oct）**。**InternVL シリーズの「軽量化 + ドメイン特化」分枝**。3 サイズ: **Mini-InternVL-1B**（Qwen2-0.5B + InternViT-300M）/ **2B**（InternLM2-1.8B + InternViT-300M）/ **4B**（Phi-3-Mini + InternViT-300M）。動的解像度・MLP プロジェクタ・Pixel Unshuffle は [[entities/internvl-1-5\|InternVL 1.5]] と同じ構造。**InternVL2-Llama3-76B の 90% 性能を 5% パラメータで**（Mini-InternVL-4B Avg 72.8 vs 76B Avg 81.4）。**Mini-InternVL-DA-1B/2B/4B**: 自律走行 / 医療 / リモセンに **統一フレームワーク**で適応、**MME-RW 自律走行で GPT-4o を +24.78 圧倒、GMAI-MMBench 医療で LLaVA-Med / RadFM / Claude3-Opus 超え、DriveLM Challenge で 2B が 26B SOTA に匹敵**。MIT + 各 LLM ライセンス。HF: `OpenGVLab/Mini-InternVL-Chat-{1B/2B/4B}-V1-5`。
- [[entities/internvit-300m]] — OpenGVLab の **InternViT-300M**。**CLIP-ViT-L-336px で初期化 + [[entities/internvl\|InternViT-6B]] から知識蒸留**（negative cosine similarity 損失、最後 K 層）した 300M 軽量視覚エンコーダ。**Mini-InternVL の視覚バックボーン**。CLIP-ViT-L 単体より OCR で +8.4 / Chart で +5.3 / InfoVQA で +8.1 圧倒。多ドメイン視覚知識（自然画像 + OCR + chart + multidisciplinary）を継承。**「VFM の知識蒸留で軽量化問題を解く」** という方向性を確立。MIT ライセンス。HF: `OpenGVLab/InternViT-300M-448px` / `InternViT-300M-448px-V2_5`（InternVL 2.5 で V2.5 へ強化）。
- [[entities/internvl-2-5]] — OpenGVLab Shanghai AI Lab の **InternVL 2.5（2024 Dec、InternVL シリーズ第 6 世代）**。**1B/2B/4B/8B/26B/38B/78B の 7 サイズスイート + InternVL2.5-Pro（非公開）**。視覚: **InternViT-300M-V2.5**（1B-8B）or **InternViT-6B-448px-V2.5**（26B+、45 層、5.5B、QK-Norm + RMSNorm）。LLM: **InternLM 2.5（1.8B/7B/20B-Chat）+ Qwen 2.5（0.5B/3B/32B/72B-Instruct）**。アーキテクチャは [[entities/internvl-1-5\|InternVL 1.5]] / 2.0 と完全に同じ ViT-MLP-LLM。**訓練 3 段階**（MLP warmup + ViT incremental learning [任意] + Full instruction tuning）+ **Progressive Scaling Strategy**（小型 LLM で ViT 訓練 → 大型 LLM 転送、Qwen2-VL の 1/12 訓練トークン）。**主要結果**: MMMU 70.1（**70% を超えた初の OS MLLM**）/ MathVista 72.3 / RefCOCO 92.3 / MMB 88.3/88.5 / MMStar 69.5 / Video-MME 72.1 / MVBench 76.4 / MLVU 75.7 / MME 2494.5。MIT + InternLM2/Qwen2.5 ライセンス。HF: `OpenGVLab/InternVL2_5-{1B/2B/4B/8B/26B/38B/78B}`。
- [[entities/internvl-3]] — OpenGVLab Shanghai AI Lab の **InternVL 3（2025 Apr、InternVL シリーズ第 7 世代）**。**1B/2B/8B/9B/14B/38B/78B の 7 サイズ**。視覚: [[entities/internvl-2-5\|InternVL 2.5]] と同じ **InternViT-300M-V2.5 / InternViT-6B-V2.5** を再利用。LLM: **Qwen2.5 base モデル**（0.5B/1.5B/7B/14B/32B/72B）+ **InternLM3-8B**（9B のみ）。**訓練パラダイム転換**: 「LLM Chat 版から MLLM 事後改造」(InternVL 1.0-2.5) → **「Native Multimodal Pre-Training」**（テキスト 50B + マルチモーダル 150B = 200B トークンの共同事前学習、1:3 比率、全層共同訓練、text-only loss + square averaging）。**新技術**: (1) **V2PE**（視覚トークンに位置インクリメント δ < 1、{1, 1/2, ..., 1/256} からランダム、$\delta=1/4$ 最良）、(2) **MPO**（DPO + BCO + LM 損失、300K 選好データ、+4.5 ポイント推論改善）、(3) **VisualPRM-8B**（Process Reward Model、Best-of-N で小型 +9.9）。**主要結果**: MMMU **72.2 で SOTA**（Claude-3.7-Sonnet 75.0 のみ上）/ MathVista 79.0（GPT-4o +19.0）/ OCRBench **906**（史上初の 900 超え）/ MME 2549.8 / MMB 89.0/88.7 / MMStar 72.5 / Video-MME 72.7/75.7 / MVBench 78.7 / MLVU 79.5 / VSI-Bench 空間推論で **InternVL3-8B (42.1) が GPT-4o (34.0) を +8.1 圧倒**。**Qwen2.5-Chat より純粋言語が強い**（+1.6〜+8.9）「マルチモーダル化で言語が強くなる」初実証。**訓練データ完全公開**（`OpenGVLab/InternVL-Data`）。MIT + Qwen2.5/InternLM3 ライセンス。**InternVL シリーズの哲学転換を象徴**。
- [[entities/mpo]] — OpenGVLab の **MPO（Mixed Preference Optimization）アルゴリズム**（2024 Nov）。MLLM の CoT 推論改善のための **3 損失混合 PO 法**: **DPO（相対選好、$w_p=0.8$）+ BCO（絶対品質、$w_q=0.2$）+ SFT loss（生成過程、$w_g=1.0$）**。学習率 5e-6、KL ペナルティ $\beta=0.1$、1 epoch、参照モデル凍結。**10 PO アルゴリズム比較で「DPO + BCO + SFT loss」が最良 CoT 性能**。**InternVL2-8B-MPO で MathVista +8.7（67.0、10× 大きい InternVL2-76B 67.2 と同等）、M3CoT +19.9、MathVision 25.7 で当時オープン SOTA**。テキスト専用ベンチでも +0.9 改善（TheoremQA +5.2、IFEval +4.1）。[[entities/internvl-3\|InternVL 3]] の Stage 3 で正式採用（+4.1～+4.5 ポイント推論改善）、**[[entities/internvl-3-5\|InternVL 3.5]] では Cascade RL の Stage 1 として進化**（MPO + GSPO の 2 段階）。HF: `OpenGVLab/InternVL2-8B-MPO`.
- [[entities/internvl-3-5]] — OpenGVLab Shanghai AI Lab の **InternVL 3.5（2025 Aug、InternVL シリーズ第 8 世代）**。**9 サイズ × dense + MoE × Flash 効率版 = 18+ モデルのスイート**。Dense: **1B/2B/4B/8B/14B/38B（Qwen3-base + InternViT-300M/6B）**。MoE: **20B-A4B（GPT-OSS-20B、OpenAI 公開モデル統合）/ 30B-A3B（Qwen3-30B-A3B）/ 241B-A28B（Qwen3-235B-A22B、最大）**。**3 つの中核イノベーション**: (1) **Cascade RL = offline RL（MPO）+ online RL（GSPO、reference 制約なし、トークン geometric mean importance ratio）**、GSPO 単独の半分の GPU 時間で +2.1 ポイント上回る、全モデルサイズで SFT 単独から +6-12 ポイント改善、(2) **ViR + ViCO**（patch ごとに 1/4 or 1/16 圧縮率を semantic richness で動的選択、視覚トークン 50% 削減で性能 99% 維持）、(3) **DvD**（ViT/MLP/ViR を vision server、LLM を language server に分離、非同期 3 段階パイプライン、896 解像度で **4.05× 推論加速**）。**主要結果**: MMMU **77.7（オープンソース新 SOTA、GPT-5 84.2 に -6.5）**、MathVista **82.7（GPT-5 81.9 +0.8）**、**VSI-Bench 69.5（GPT-5 37.5 を +32 圧倒）**、WildVision **82.8（GPT-4o 超え）**、WindowsAgentArena で GPT-4o 圧倒、Cascade RL 全サイズで +6-12 改善、Parallel Thinking でさらに +1-9 改善。**Qwen3-base 比で言語 16 ベンチ中 15 で上回る**（InternVL 3 の発見継続）。**GPT-5 との Aggregate 差 3.9%** という当時オープンソース最小。MIT + Qwen3/GPT-OSS ライセンス。HF: `OpenGVLab/InternVL3_5-{1B/2B/4B/8B/14B/38B/20B-A4B/30B-A3B/241B-A28B}` + Flash 版。**InternVL シリーズ現時点最新版**。
- [[entities/sdxl]] — Stability AI Applied Research の **SDXL（Stable Diffusion XL、2023 Jul, arXiv:2307.01952、wiki 初の拡散モデル系統）**。Rombach lab（LDM 原論文の著者ら）の Stable Diffusion 第 3 世代（SD 1.x → 2.x → **SDXL**）。**2 段階パイプライン（base + refiner）**、出力 1024×1024 中心の 40 アスペクト比。**Base UNet 2.6B**（SD 1.x の 860M から 3×、transformer blocks [0,2,10]、channel mult [1,2,4]、最低レベル削除）+ **二重テキスト・エンコーダ**（[[entities/clip\|CLIP]] ViT-L + OpenCLIP ViT-bigG = 817M、context dim 2048、ペナルティメイト連結 + pooled OpenCLIP を timestep に加算）+ **改良 VAE**（ゼロから訓練、batch 256、EMA、PSNR 24.7 / SSIM 0.73 / LPIPS 0.88 / rFID 4.4 で SD 1.x/2.x VAE 全凌駕）+ **Refinement Model**（同じ潜在空間の別 LDM、最初の 200 ノイズ・スケール特化、SDEdit ベース）。**3 つの新規条件付け**: **size-conditioning**（フーリエ特徴で元 H, W、39% データ廃棄問題解決、ImageNet FID 36.53）/ **crop-conditioning**（クロップ座標 $(c_{\text{top}}, c_{\text{left}})$、物体切れ解決）/ **multi-aspect conditioning**（40 アスペクト比 0.25-4.0、target size 埋め込み）—すべてフーリエ特徴で timestep 埋め込みに加算。**訓練**: 256² 600k → 512² 200k → multi-aspect ~1024²、batch 2048、offset-noise 0.05、離散時間 1000 ステップ、CFG（10% null）。**主要結果**: ユーザ研究で **SDXL w/ refinement 48.44% / base 36.93% / SD 1.5 7.91% / SD 2.1 6.71%**（refiner 付きで SD 2.1 比 7.2× 選好）。**Midjourney v5.1 を 54.9% で凌駕**（17,153 投票、PartiPrompts P2、AWS GroundTruth）、6 カテゴリ中 4 / 10 チャレンジ中 7 で凌駕または同等。DALL-E 2 / Bing Image Creator / DeepFloyd IF を質的比較で凌駕。**FID/CLIP 指標の信頼性問題を学術的に裏付け**（SDXL の FID は SD 1.5/2.1 より悪いが人間評価で明確に好まれる、Pick-a-pic 発見の再確認）。**CreativeML Open RAIL++-M License**（商用可、Apache 2.0 風）、HF: `stabilityai/stable-diffusion-xl-base-1.0` + `stabilityai/stable-diffusion-xl-refiner-1.0`、コード `github.com/Stability-AI/generative-models`。**Computer Vision wiki 初の拡散モデル系統 ingest**として [[concepts/diffusion-model]] の主要実装例、認識系（SAM、DINOv3、CLIP）に並ぶ生成系基盤モデル代表。**後継**: SDXL Turbo（蒸留 1 ステップ）、Stable Cascade（Würstchen）、SD3（DiT + T5-XXL）、SD3.5、FLUX.1（Rombach ら Black Forest Labs 独立、12B Hybrid MM-DiT）。**3 つの新規条件付けと refiner パラダイムは後続拡散モデル（SD3/FLUX/Imagen 2 等）で標準化**。**弱点**: 人間の手、完全な写真現実性に届かず、社会的バイアス、concept bleeding（CLIP 単一トークン圧縮起因の属性バインディング失敗）、長文テキスト・レンダリング、二段階推論コスト、offset-noise 必須、DiT 系試行は即時利益なし、訓練データ非公開。
- [[entities/deepseek-ocr]] — DeepSeek-AI の **DeepSeek-OCR（2025 Oct, arXiv:2510.18234、wiki 初の DeepSeek-AI モデル）**。**OCR / 文書理解特化型 MLLM**、Haoran Wei（GOT-OCR2.0 の著者）+ Yaofeng Sun + Yukun Li。**「視覚モダリティをテキスト圧縮媒体として活用する」新パラダイム**。**DeepEncoder**: [[entities/sam\|SAM-base]] (80M, window attention) + 2 層 ConvNet (16× 圧縮) + [[entities/clip\|CLIP-large]] (300M, global attention) = 約 380M。**DeepSeek-3B-MoE-A570M デコーダ**: 64 routed + 6 活性化 + 2 shared、推論時 570M 活性。**6 解像度モード**: Tiny 64 / Small 100 / Base 256 / Large 400 / Gundam 動的 / Gundam-M。**Fox ベンチで 10× 圧縮 97% / 20× 圧縮 60%**。**OmniDocBench 編集距離 0.083 SOTA**（Gundam 795 トークン、MinerU2.0 6790 トークン 0.133 を 8.5× 効率で凌駕）、100 トークンで GOT-OCR2.0 凌駕。**1 台 A100-40G で 200K+ ページ/日**。**Memory Forgetting Mechanism**: 古い対話を画像化・縮小して LLM 長文脈問題を解決する応用提案。**訓練データ**: OCR 1.0 70%（30M+ PDF・100 言語、シーン OCR 中英各 10M）+ OCR 2.0（10M チャート + 5M 化学式 + 1M 平面幾何）+ 一般視覚 20% + テキスト 10%。**訓練**: 20 ノード × 8× A100-40G = 160 GPU、PP=4 + DP=40、70-90B トークン/日。**オープンソース** (GitHub)。同著者の系譜: Vary → GOT-OCR2.0 → DeepSeek-OCR。**Qwen 系 / InternVL 系 / Google 系 (Gemma) に続く第 4 の中国系 AI ラボ系譜**。**弱点**: OCR 特化（MMMU 報告なし）、SFT なし、Newspapers 0.645、Memory Forgetting は概念実証のみ、DeepSeek-VL2 との分業不明確。
- [[entities/gemma-3]] — Google DeepMind の **Gemma 3（2025 Mar, arXiv:2503.19786、Gemma 系列第 3 世代）**。**「wiki 初の Google 系オープン MLLM、軽量 + 高性能のバランス代表」**。**1B/4B/12B/27B の 4 サイズ × PT/IT = 8 公開モデル**。**1B は text-only、4B/12B/27B はマルチモーダル**（[[entities/siglip\|SigLIP]] 400M variant を共有 + 訓練中凍結、896² 固定、4×4 average pooling で **256 トークン固定圧縮**）。**Pan & Scan (P&S)**: 推論時のみの LLaVA 風タイル分割で非正方形・高解像度画像対応（27B で InfoVQA +17.0 / DocVQA +4.8）。**5:1 local:global attention + sliding window 1024**（KV キャッシュ・オーバーヘッド 60% → <15% 削減）。**128K 文脈**（1B は 32K、global RoPE base 10k→**1M**、32K→128K スケーリング係数 8）。**QK-norm**（Gemma 2 soft-capping 置換）+ GQA + RMSNorm。**QAT**: per-channel int4 / per-block int4 / SFP8 の 3 形式、**27B Int4 で 14.1 GB**（消費者 GPU 可）。事前学習: 27B 14T / 12B 12T / 4B 4T / 1B 2T トークン、Gemini 2.0 と同じ 262k 語彙 SentencePiece、**全モデル知識蒸留**（256 ロジット/トークン）。事後学習: BOND + WARM + WARP RL 改善版 + 重み平均報酬モデル。**主要結果**（27B IT）: **LMSys Chatbot Arena Elo 1338（rank 9）** で DeepSeek-V3 / Llama-3.1-405B / Qwen2.5-72B を凌駕、MMLU-Pro 67.5、**MATH 89.0**、MMMU val **64.9**、DocVQA 86.6（w/ P&S 90.4）、InfoVQA 70.6（w/ P&S 76.4）、MathVista 67.6。**PaliGemma 2 27B を文書理解で凌駕** + **4B/12B は 10× 安価転送**。**Gemma3-4B-IT が Gemma2-27B-IT 匹敵、Gemma3-27B-IT が Gemini-1.5-Pro 同等**。**Gemma License（Apache 2.0 風、商用可）**。HF: `google/gemma-3-{1b,4b,12b,27b}-{pt,it}`。**Memorization** が全前世代より桁違いに低く、個人情報も観察されず。**[[questions/vit-dynamic-resolution-evolution]] のタイル分割路線の最新形** + **KV キャッシュ最適化の新標準**。**Qwen 系 / InternVL 系と並ぶ第 3 の主要系譜（Google 系）**。**弱点**: 固定 896² + P&S の保守性、256 トークン圧縮、128K で RULER 66.0（Qwen3-VL 1M YaRN 99.5% に大きく劣る）、動画 16 frames 制約、**MoE 不採用**（InternVL 3.5 / Qwen3-VL と対照的）、訓練データ非公開。
- [[entities/qwen3-5-omni]] — Alibaba Group / Qwen Team の **Qwen3.5-Omni（2026, arXiv:2604.15804、Qwen-Omni シリーズ最新版）**。**「テキスト + 画像 + 音声 + 動画の完全 omnimodal 統合 + リアルタイム音声対話エージェント」**。**Plus（フラッグシップ）+ Flash の 2 バリアント**、**API のみ公開**。**Thinker-Talker 構造**: Thinker（Hybrid MoE Transformer + Qwen3.5 ベース + GDN）がテキスト生成、Talker（Hybrid MoE Transformer + MTP + Code2Wav）がストリーミング音声トークン生成。**Audio Encoder**: **AuT（ゼロから 4000 万時間で学習、6.25Hz、20+ 言語、中英多言語 3.5:3.5:3、動的注意ウィンドウ）**。**Vision Encoder**: SigLIP2（Qwen3.5 共有）。**7 主要進化**: (1) Hybrid MoE 双方、(2) **256K ネイティブ**（10 時間音声 / 400 秒動画 / 1 億時間 AV データ）、(3) AuT、(4) マルチコードブック RVQ で単一フレーム即時合成、(5) **ARIA (Adaptive Rate Interleave Alignment)**（MFA 不要、適応的レート制約でテキスト-音声整合動的化）、(6) **音声入力 113 言語+方言 / 音声出力 36 / テキスト 201**、(7) **Audio-Visual Vibe Coding**（音声-視覚命令→直接コード生成、出現能力）。**3 段階事前学習**: S1 Encoder Alignment + S2 General（~4T、seq 32K）+ S3 Long Context（seq 262K）。**事後学習**: Thinker 3 段階（Specialist Distillation + **On-Policy Distillation (OPD)** + Interaction-Aligned RL）+ Talker 4 段階（General + Long-Context CPT 64K + DPO/GSPO RL + Speaker Fine-tuning）。**First-Packet Latency**: Flash 235ms/426ms、Plus 435ms/651ms（音声/動画入力）。**主要結果**: **215 音声・AV サブタスクで SOTA**、Gemini-3.1 Pro を音声タスク主要部で凌駕、AV 理解で同等。MMAU **82.2** / MMSU **82.8** / RUL-MuchoMusic **72.4 (+12.8)** / VoiceBench **93.1 (+4.2)** / FLEURS ASR **6.55 WER** / Librispeech clean **1.11 WER** / **Kespeech 3.46 WER (Gemini 23.67 = 6.8× 改善)** / **DailyOmni 84.6 SOTA** / Qualcomm IVD **68.5 (+2.3)** / Omni-Cloze **64.8 (+7.6)** / **SEED-TTS test-en 1.26 WER SOTA** / クロス言語音声生成 12/10 SOTA / FLEURS 60 言語 ASR 平均 **6.6 WER**。視覚: 同サイズ純粋 VL と同等、**動画 6/6 ベンチで上回る**（共同 video-audio 学習効果、RealWorldQA +5.0）。Qwen ファミリーの **「Omni 統合路線」**（vs [[entities/qwen3-vl|Qwen3-VL]] の「VL 純化路線」）として並列発展。**弱点**: OmniGAIA で Gemini に -11.7、Plus/Flash 2 バリアントのみ、API のみ公開（モデル重み非公開）、訓練データ非公開、4000 万時間音声学習は独自データ依存。
- [[entities/qwen3-vl]] — Alibaba Group / Qwen Team の **Qwen3-VL（2025 Nov, arXiv:2511.21631、Qwen-VL シリーズ第 4 世代）**。**「Qwen ファミリーが商用最先端と完全に肩を並べた決定的瞬間」**。**6 サイズ × 2 バリアント = 12 公開モデル**: dense **2B/4B/8B/32B** + MoE **30B-A3B / 235B-A22B（フラッグシップ）** × **Instruct（non-thinking）+ Thinking（長 CoT 推論）**。**Vision Encoder**: **SigLIP-2 から継続学習**（[[entities/qwen2-5-vl\|Qwen2.5-VL]] のゼロから路線放棄）、SigLIP2-SO-400M（235B/32B/30B-A3B/8B）/ SigLIP2-Large 300M（2B/4B）、CoMP 方法論で 2D-RoPE + 絶対位置補間。**Vision-Language Merger**: 2 層 MLP で 2×2 → 1 トークン圧縮 + DeepStack 用専用マージャ。**3 主要構造革新**: (1) **Interleaved MRoPE**（t/h/w を埋め込み次元に交互配置、周波数スペクトル均衡化）、(2) **DeepStack**（ViT 中間 3 層特徴を LLM 最初 3 層に注入、追加文脈長なし、アブレ +1.3 ポイント平均）、(3) **テキスト・ベース時間整合**（`<3.0 seconds>` 明示的タイムスタンプ・トークン、秒形式 + HMS 形式）。**256K ネイティブ文脈**（YaRN 拡張で **1M トークン = 2 時間動画**まで外挿）。**4 段階事前学習**（S0 Merger 67B + S1 All 1T 8K + S2 All 1T 32K + **S3 Ultra-Long-Context 100B 262K** = 累積 ~2.2T）。**事後学習**: SFT（120 万、32K → 256K、non-thinking + thinking 分岐）+ Strong-to-Weak Distillation（テキスト専用 LLM 蒸留、Off-policy → On-policy KL）+ RL（**SAPO アルゴリズム**、Reasoning RL + General RL、約 30K クエリ）+ **Thinking with Images**（2 段階、Qwen2.5-VL-32B で 10K → 蒸留 120K、3 報酬: Answer Accuracy + Multi-Turn Reasoning + Tool-Calling）。**平方根再重み付け**で text/multimodal バランス改善。**事前学習データ 8 カテゴリ**（Image Caption + Interleaved 256K mergedシーケンス / Knowledge 12+ カテゴリ importance-based sampling / OCR 多言語 **39 言語** / Grounding [0,1000] 正規化座標復帰 + Grounding DINO 合成 / 9-DoF 3D bbox Omni3D / Multimodal Code UI→HTML/SVG / Video Length-Adaptive Sampling + Dense Caption / STEM 6000 万 K-12 + 1200 万 CoT + **Multimodal Necessity Filtering** / Agent GUI + Function Calling + Search）。文書フォーマット: **QwenVL-HTML**（要素レベル bbox）+ **QwenVL-Markdown**（表は LaTeX 符号化）。**主要結果**（235B-A22B）: **MathVista 85.8 / MathVision 74.6 / MathVerse_mini 85.0 SOTA**、MMMU 80.6（GPT-5 84.2 -3.6）、**OCRBench 920 / OCRBench_v2 zh 63.5 SOTA**（GPT-5 +25.8 圧倒）、OmniDocBench en 0.143 SOTA、MMLongBench-Doc 57.0 SOTA、MUIRBENCH 80.1 SOTA、CountBench 93.7、**ODinW-13 48.6**、VSI-Bench 62.7 / EmbSpatialBench 84.3 / RefSpatial 69.9、**Charades-STA mIoU 64.8 SOTA**、V\* with tools 93.7+、**ScreenSpot Pro 62.0**（Qwen2.5-VL から +18.4）、**OSWorld 38.1**（Qwen2.5-VL 8.83 から **4.3× 飛躍**）、**AndroidWorld 63.7**（Qwen2.5-VL から +28.7）、OSWorldG 68.3、WindowsAA 32.1。**Needle-in-a-Haystack**: 30 分動画（256K）完全 100% / YaRN 拡張で **1M（2 時間動画）99.5%**。**純粋テキスト**: AIME-25 **+4.4** / HMMT-25 +2.0 / LiveCodeBench v6 +2.5 vs Qwen3 純粋 LLM、「マルチモーダル化で言語が強くなる」継続実証。**Apache 2.0**。HF: `Qwen/Qwen3-VL-{2B/4B/8B/32B/30B-A3B/235B-A22B}-{Instruct/Thinking}`。**インフラ**: Alibaba Cloud PAI-Lingjun、Megatron-LM、5D 並列（TP+PP+CP+EP+ZeRO-1 DP）、最大 10,000 GPU、vLLM / SGLang 推論。**Interleaved MRoPE + DeepStack + テキスト・タイムスタンプ + MoE + thinking モード**は MLLM 業界の新標準。**弱点**: MMMU で GPT-5 -3.6、We-Math で Gemini -5.8、Video-MME / MLVU で商用にやや劣る、OSWorld で Claude に -6.3、訓練データ非公開、6×2=12 モデル管理コスト大、MoE 235B は複数 GPU 必須。
- [[entities/qwen2-5-vl]] — Alibaba Group / Qwen Team の **Qwen2.5-VL（2025 Feb, arXiv:2502.13923、Qwen-VL シリーズ第 3 世代）**。**「視覚言語モデルから視覚エージェントへの転換点」**。**3B / 7B / 72B の 3 サイズ**（全サイズで同一 ViT 共有、LLM は Qwen2.5-2048H/3584H/8192H、KV 2/4/8、Embedding Tying 3B のみ）。**Vision Encoder**: 32 層 ViT を **ゼロから学習**（DataComp + 社内データ初期化）、**32 層中 4 層のみ完全自己注意**（インデックス {7,15,23,31}）、残り 28 層は **112×112 ウィンドウ・アテンション**（8×8 パッチ）、計算量を 2 次 → 線形化、**RMSNorm + SwiGLU**、patch 14、動画は連続 2 フレーム grouping（3D パッチ分割）。**Vision-Language Merger**: MLP 2 層で隣接 4 パッチを 1 トークンに圧縮、In 1280 → Out 2048/3584/8192。**4 主要技術**: (1) Window Attention、(2) **MRoPE Aligned to Absolute Time**（temporal ID を絶対秒数に揃える、異なる FPS の動画で一貫した時間整合）、(3) **Native Dynamic Resolution and Frame Rate**（座標は入力画像実次元で表現、動的 FPS サンプリング、hmsf 形式タイムスタンプ）、(4) **事前学習データ 1.2T → 4.1T トークンへ拡張**。**3 段階事前学習**: Visual（1.5T、ViT のみ、seq 8K）+ Multimodal（2T、ViT & LLM、seq 8K）+ Long-Context（0.6T、ViT & LLM、seq 32K）。**2 段階事後学習**: SFT 200 万（純粋テキスト 50% + マルチモーダル 50%、ChatML、Qwen2-VL-Instag で 8 ドメイン × 30 サブカテゴリ階層分類 + 棄却サンプリング）+ DPO（画像-テキスト + 純粋テキスト、ViT 凍結）。**QwenVL HTML フォーマット**（独自貢献）: data-bbox 属性 + HTML タグでレイアウト・表・チャート・数式・楽譜（ABC notation）・化学式（SMILES）を統一表現。**主要結果**（72B）: **MMMU 70.2**（初の 70 突破、Qwen2-VL から +5.7、GPT-4o 69.1 超え）/ **MathVista 74.8**（GPT-4o +11.0）/ **MATH-Vision 38.1**（Qwen2-VL から +12.2）/ MathVerse 57.6 / MMBench-EN **88.6 SOTA** / MMBench-V1.1-EN 88.4 / MMStar **70.8 SOTA** / MMVet **76.2 SOTA** / MME-RealWorld **63.2 SOTA** / DocVQA **96.4** / InfoVQA **87.3** / OCRBench **885** / OCRBench_v2 en/zh **61.5/63.7**（Gemini 1.5-Pro 比 en +9.6 / zh +20.6 圧倒）/ CC-OCR **79.8** / OmniDocBench edit en **0.226 SOTA** / CharXiv DQ **87.4** / SEED-Bench-2-Plus **73.0** / **CountBench 93.6 SOTA**（GPT-4o +5.7、Molmo-72B +2.4）/ RefCOCO 92.7（InternVL2.5-78B 93.7 にやや後れ）/ ODinW 43.1（オープン語彙検出）/ PointGrounding 67.5 / MVBench **70.4** / MMBench-Video **2.02** / LVBench **47.3**（GPT-4o 30.8 +16.5）/ EgoSchema **76.2** / MLVU **74.6**（GPT-4o 64.6 +10.0）/ TempCompass 74.8 / **Charades-STA mIoU 50.9 SOTA**（時間グラウンディング、GPT-4o 35.7 +15.2）。**GUI エージェント**: ScreenSpot Pro **43.6**（Qwen2-VL 1.6% から **+42 ポイント、27× 飛躍**、Aguvis-72B 23.6 圧倒）/ Android Control High EM **67.36** / Low EM **93.7** / AndroidWorld SR **35%**（GPT-4o 34.5% 超え）/ MobileMiniWob++ **68%** / OSWorld 8.83（Claude 14.90 に -6.07）。**純粋テキスト**: Qwen2.5-72B（純粋 LLM）とほぼ同等を維持（LiveBench / MultiPL-E / IFEval で上回る、マルチモーダル化で言語が強くなる現象）。**Apache 2.0**（3B/7B）/ Qwen License（72B）。HF: `Qwen/Qwen2.5-VL-{3B/7B/72B}-Instruct`。**Window Attention + MRoPE absolute time + 統一文書 HTML + GUI agent パラダイム**は MLLM 業界の新標準、後の Qwen3-VL の基盤。
- [[entities/qwen2-vl]] — Alibaba Group / Qwen Team の **Qwen2-VL（2024 Sept, arXiv:2409.12191、Qwen-VL シリーズ第 2 世代）**。**Naive Dynamic Resolution + M-RoPE で「任意解像度」を実現**。**2B / 7B / 72B の 3 サイズ**（全サイズで **675M ViT 共有**、DFN 初期化 + 絶対位置を 2D-RoPE に置換、LLM は Qwen2 1.5B/7.6B/72B）。**Naive Dynamic Resolution**: ViT を任意解像度対応に + MLP で隣接 2×2 トークンを 1 トークンに圧縮（224² → 66 トークン）、`min_pixels=100×28²` / `max_pixels=16384×28²`。**M-RoPE**: temporal/height/width の 3 成分、テキストは 1D-RoPE 等価、画像は temporal 固定 / 動画は temporal 増分、位置 ID 値を小さく抑えて長文脈外挿に有利。**統一画像/動画処理**: 動画 2 fps + 深さ 2 の 3D 畳み込み（3D チューブ）、動画あたり 16K トークン上限、**学習 16K → 推論 80K まで外挿可能**。**文脈長 32K**。**3 段階学習** + 累積 **1.4T トークン**（Stage1: LLM 凍結で 600B / Stage2: 全凍結解除で +800B / Stage3: ViT 凍結 SFT、テキスト・トークンのみ監督）。特殊トークン: `<|vision_start/end|>` / `<|box_start/end|>` / `<|object_ref_start/end|>` / `<|im_start/end|>`、座標は **[0, 1000) 正規化文字列**。**主要結果**（72B）: DocVQA **96.5** / InfoVQA **84.5** / AI2D **88.1** / OCRBench **877** / MTVQA **30.9** / VCR-zh **65.4** / RealWorldQA **77.8** / MMBench-CN **86.6 SOTA** / MathVista **70.5** / MME **2482.7** / MMStar **68.3** / MMVet **74.0** / RefCOCO val **93.2 SOTA** / MVBench **73.6** / EgoSchema **77.9**（GPT-4o +5.7）/ Video-MME 71.2/77.8 / 多言語 OCR 8 言語中 7 で GPT-4o 超え / Function Calling EM 53.2 + AITZ UI EM 72.1（GPT-4o の 2 倍）+ Number Line/EZPoint 100%。**弱点**: MMMU 64.5（GPT-4o 69.1 に -4.6）、MathVision 25.9（-4.5）、R2R 51.7（専門 79.0 に -27.3、3D 空間モデリング）、アラビア語 OCR -5.2、動画 16K 上限。**Apache 2.0**（2B/7B）/ Tongyi Qianwen License（72B）。HF: `Qwen/Qwen2-VL-{2B/7B/72B}-Instruct`。**Naive Dynamic Resolution + M-RoPE が MLLM 業界の事実上の標準**として広く採用、後の Qwen2.5-VL / Qwen3-VL の基盤。
- [[entities/qwen-vl]] — Alibaba Group の **Qwen-VL / Qwen-VL-Chat（2023 Aug, arXiv:2308.12966）**。**Qwen-VL シリーズの初代モデル**。**Qwen-7B（中英バイリンガル LLM）+ OpenCLIP ViT-bigG（1.9B、パッチ 14）+ Position-aware VL Adapter（0.08B、単層クロスアテンション + 256 学習可能クエリ + 2D 絶対位置エンコーディング）= 9.6B**。**3 段階学習**: Stage1（224²、LLM 凍結、1.4B 画像-テキスト対、50K ステップ）→ Stage2（448²、全凍結解除、7 タスク同時 = Captioning 19.7M + VQA 3.6M + Grounding 3.5M + Ref Grounding 8.7M + Grounded Cap 8.7M + OCR 24.8M + Pure-text 7.8M、19K ステップ）→ Stage3（SFT 350K、ViT 凍結、Qwen-VL-Chat を得る、ChatML 形式）。**4 種特殊トークン**: `<img></img>` / `<box></box>` / `<ref></ref>`、座標は **[0, 1000) 正規化文字列**（位置語彙不要）。**主要結果**: Flickr30K (0-shot) **85.8 CIDEr** / VQAv2 **79.5** / OKVQA **58.6** / GQA **59.3** / TextVQA **63.8** / DocVQA 65.1 / OCR-VQA **75.7** / RefCOCO val **89.36** / RefCOCO+ val **83.12** / RefCOCOg val **85.58** / GRIT refexp **78.22** / TouchStone En **645.2** + Cn **401.2** / SEED-Bench All **58.2** / MME Perception **1487.58** + Cognition **360.71**。**Qwen-VL License**（オープンソース、研究・商用可）。HF: `Qwen/Qwen-VL` / `Qwen/Qwen-VL-Chat`。後継は Qwen2-VL（2024.09、2B/7B/72B）/ Qwen2.5-VL（2025）。
- [[entities/mmpr]] — OpenGVLab の **MMPR（MultiModal PReference dataset）**（2024 Nov）。**約 3M サンプル**（正解あり 2.5M + 正解なし 750K）の MLLM 用選好データセット。**6 ドメイン × 18 データソース**（General VQA / Science / Chart / Mathematics / OCR / Document）。**2 つのパイプライン**: (1) **正誤判定ベース**（最大 32 解答サンプリング、最大 15 ペア / クエリ、temperature 1.0）、(2) **DropoutNTP**（応答を半分で切り詰めて画像なしで補完、Dropout Ratio 0.5 最適）。**RLAIF-V の 57.5% コスト**（選好ペアあたり 571.2 vs 992.7 トークン）。**InternVL 3 で MMPR v1.2（拡張版）使用、Stage 3 で 300K ペアを抽出**。VisualPRM400K（VisualPRM-8B 訓練データ）も MMPR v1.2 ベース。HF: `OpenGVLab/MMPR-v1.2`、MIT ライセンス。
- [[entities/siglip]] — Google の SigLIP / SigLIP 2。CLIP の sigmoid 損失改良（v1）＋ 全部入りレシピ統合（v2: LocCa decoder + 自己蒸留＋マスク予測 + ACID + 多言語 + NaFlex）。DINOv3 の主要競合（弱教師あり）。
- [[entities/sam]] — SAM（Meta, 2023）。CV における初の本格的セグメンテーション基盤モデル。ViT-B/L/H 3 種、Apache 2.0、データエンジンで 1.1B マスクを使い訓練。
- [[entities/sam-2]] — SAM 2（Meta, 2024）。SAM の動画拡張版。Hiera 画像エンコーダ + streaming memory、Hiera-T/S/B+/L 4 種、Apache 2.0。画像でも SAM v1 比 6× 高速。
- [[entities/sam-3]] — SAM 3（Meta Superintelligence Labs, 2025）。PCS タスクを導入、PE backbone + DETR detector + presence head + SAM 2 tracker。Apache 2.0 想定。SA-Co/Gold で人間性能の 74% 達成。
- [[entities/hiera]] — Hiera（Meta, 2023）。MAE 互換の階層型 ViT、shifted window や RPB を削除したシンプル設計。SAM 2 の画像エンコーダ。
- [[entities/simclr]] — SimCLR（Google Brain, 2020）。対比 SSL の標準フレームワーク。4 コンポーネント設計と体系的アブレーションで 2020 年 SOTA を確立。
- [[entities/moco]] — MoCo（FAIR, CVPR 2020）。**Momentum encoder（EMA 更新の target network）+ queue（FIFO メモリバンク、65k 負例）** で対比 SSL を大バッチ依存から解放した先駆け。v1 (2019) → v2 (SimCLR の MLP head 統合、ImageNet linear 71.1%) → v3 (2021、ViT 対応、76.7%)。**[[entities/byol]] / [[entities/dino]] の momentum target / teacher の直接の源流**。本 wiki では source ページ未取り込み（ingest 候補）、概要ページのみ。
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
- [[entities/mmpr]] — MMPR（MultiModal PReference dataset、2024 Nov）。**約 3M サンプル**（正解あり 2.5M + 正解なし 750K）の MLLM 用選好データセット。**6 ドメイン × 18 データソース**。DropoutNTP + 正誤判定ベースの 2 パイプラインで構築。MMPR v1.2（拡張版）が InternVL 3 の MPO Stage 3 で使用される。HF: `OpenGVLab/MMPR-v1.2`。

### People

（まだありません）

### Organizations / Benchmarks

（まだありません）

## Questions

- [[questions/vit-dynamic-resolution-evolution]] — ViT における解像度処理の進化（固定 224² → タイル分割 → ViT 改造で任意解像度）。原因 4 つ（学習可能絶対位置埋め込み / データセット規格化 / 計算 O(n²) / 帰納バイアス弱）と進化 3 圧力（MLLM タスク要求 / RoPE / ハードウェア成熟）、4 フェーズ年表、タイル路線 vs ViT 改造路線、ViT 初期化戦略の揺れ動き（DFN → ゼロから → SigLIP-2 継続学習）を含む。
- [[questions/large-scale-pretraining-series]] — 大規模事前学習の **5 系列**を整理。**ユーザの「CLIP 系 + DINO 系」の認識が「半分正しい」**ことを示し、**用語的に WSL（CLIP 系）と SSL（DINO 系）の区別**を冒頭で明示。実際には **①WSL 対比型（CLIP/SigLIP/PE）/ ②自己蒸留 SSL（DINO/v2/v3）/ ③MIM 系（MAE/BEiT）/ ④古典対比 SSL（SimCLR/MoCo/BYOL/SwAV、pre-ViT）/ ⑤ハイブリッド（iBOT/DINOv2/v3）**の 5 系列が独立して存在。各系列の系譜・アーキテクチャ進化・データ大規模化（4 億 → 100 億 / 1.3M → 17 億）・革新的工夫を網羅。**2024-2025 の系列融合（SigLIP 2 が SSL 借用 / PE が中間層活用 / DINOv3 が RoPE 採用）**、**MLLM の Vision encoder は独立系列ではなく CLIP/SigLIP 系の継承・改造**であることを論証。Qwen-VL 系 / InternVL 系 / Gemma 3 / DeepSeek-OCR すべての視覚エンコーダの出自を一覧化。次の ingest 候補（ALIGN / MetaCLIP / DFN / BEiT / MoCo 等）も提示。

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
| InternVL | Internal Vision-Language model（OpenGVLab, CVPR 2024）、6B 視覚 + 8B 言語ミドルウェア | [[sources/internvl]] / [[entities/internvl]] |
| InternViT-6B | 5.9B vanilla ViT（width 3200 / depth 48 / MLP 12800 / 25 heads）、InternVL の視覚エンコーダ | [[entities/internvl]] |
| QLLaMA | Query LLaMA（8B = 多言語 LLaMA-7B + 1B 新規 96 queries+cross-attn）、InternVL の言語ミドルウェア | [[sources/internvl]] |
| InternVL-C | InternVL Contrastive（InternViT のみ + QLLaMA [EOS]、対比モード） | [[entities/internvl]] |
| InternVL-G | InternVL Generative（InternViT + QLLaMA 96 queries、対比 + 生成統合） | [[entities/internvl]] |
| InternVL-Chat | InternVL + LLM デコーダ（MLP or QLLaMA glue + Vicuna/InternLM） | [[entities/internvl]] |
| QFormer | Querying Transformer（BLIP-2 の 188M glue layer、BERT-base 初期化、32 queries） | [[sources/internvl]] |
| ITC / ITM / ITG | Image-Text Contrastive / Matching / Grounded Generation（BLIP-2 系 3 損失） | [[sources/internvl]] |
| LLaMA / Vicuna / InternLM | LLM 系（Meta / lmsys / Shanghai AI Lab）、InternVL の言語デコーダ候補 | [[entities/internvl]] |
| BLIP / BLIP-2 / InstructBLIP | Salesforce の VLM 系列（QFormer の祖）、InternVL の比較相手 | [[sources/internvl]] |
| LLaVA / LLaVA-1.5 / LLaVA-NeXT | Visual Instruction Tuning の代表作（Liu et al., 2023）、InternVL-Chat の直接比較 | [[sources/internvl]] |
| MME | Multimodal Evaluation（14 サブタスクの総合認知ベンチマーク、Fu et al., 2023） | [[sources/internvl]] |
| POPE | Polling-based Object Probing Evaluation（物体ハルシネーション、Li et al., 2023） | [[sources/internvl]] |
| Tiny LVLM | LVLM 用軽量ベンチマーク | [[sources/internvl]] |
| VQAv2 / GQA / VizWiz / TextVQA / NoCaps | VLLM 評価標準セット | [[sources/internvl]] |
| OK-VQA / A-OK-VQA / IconQA / AI2D | 外部知識・図表 VQA | [[sources/internvl]] |
| OCR-VQA / ChartQA / DocVQA / InfoVQA / ST-VQA / LLaVAR | OCR/文書 VQA | [[sources/internvl]] |
| LAION-COCO / COYO / Wukong / CC3M / CC12M / SBU | InternVL Stage 1/2 訓練用 Web 画像-テキスト対 | [[sources/internvl]] |
| LAION-en / LAION-multi | 公開 Web 画像-テキスト（LAION-5B 系、英語/多言語） | [[sources/internvl]] |
| ViT-22B | Google の 21.7B ViT（JFT-3B 訓練、2023）、InternViT-6B の主要比較相手 | [[sources/internvl]] |
| ViT-G / ViT-e / EVA-02-ViT-E / ViT-6.5B | 大型 ViT 系列、InternViT-6B のスケール比較 | [[sources/internvl]] |
| CoCa / LiT-22B | Google の対比 + 生成 / locked image チューニング、JFT-3B 系 | [[sources/internvl]] |
| Qwen-VL / Qwen-VL-Chat | Alibaba の VLM、InternVL-Chat と同時期の競合 | [[sources/internvl]] |
| Flamingo / IDEFICS / KOSMOS-2 / Shikra | 初期 VLLM 系（Cross-Attn / Linear glue） | [[sources/internvl]] |
| Emu / Emu-I / DreamLLM / PaLI-X-55B | 生成 VLLM 系 | [[sources/internvl]] |
| glue layer | 視覚エンコーダと LLM をつなぐ層（QFormer/MLP/Linear/Cross-Attn/QLLaMA 等の総称） | [[sources/internvl]] |
| VLLM | Vision Large Language Model（画像入力可能な LLM、LLaVA/BLIP-2 等の総称） | [[sources/internvl]] |
| AGI | Artificial General Intelligence（汎用人工知能） | [[sources/internvl]] |
| JFT-3B | Google 非公開の 3B 弱教師あり分類データ（ViT-22B / CoCa / LiT-22B が使用） | [[sources/internvl]] |
| InternVL 1.5 | InternVL シリーズ第 4 世代（2024 April）、26B、GPT-4V 対抗の MLLM | [[sources/internvl-1-5]] / [[entities/internvl-1-5]] |
| InternVL 1.2 | InternVL の第 3 世代（2024 Feb）、40B、QLLaMA 廃止、Yi-34B 連携 | [[entities/internvl-1-5]] |
| InternViT-6B-448px-V1.2 | InternViT 第 2 版（45 層、固定 448）、Yi-34B 連携 | [[entities/internvl-1-5]] |
| InternViT-6B-448px-V1.5 | InternViT 第 3 版（45 層、dynamic 448）、InternLM2-20B 連携 | [[entities/internvl-1-5]] |
| InternLM2-20B-Chat | Shanghai AI Lab の独自 LLM 20B Chat 版（InternVL 1.5 の LLM） | [[entities/internvl-1-5]] |
| Pixel Shuffle | 空間 → チャンネル変換（超解像由来、InternVL 1.5 で visual token 1/4 圧縮） | [[sources/internvl-1-5]] |
| Dynamic High-Resolution | 動的高解像度（InternVL 1.5 の中心的貢献、35 種アスペクト比） | [[sources/internvl-1-5]] |
| 35 種アスペクト比 | 1-12 タイルの全組合せ（1:1, 1:2, ..., 2:6） | [[sources/internvl-1-5]] |
| Thumbnail | グローバル文脈用 448×448 サムネイル（タイルに併設） | [[sources/internvl-1-5]] |
| Continuous Learning（VFM） | InternViT を継続事前学習で強化する戦略（V1.0 → V1.5） | [[sources/internvl-1-5]] |
| ViT-MLP-LLM | LLaVA 系の標準アーキテクチャ（QFormer ではなく MLP プロジェクタ） | [[sources/internvl-1-5]] |
| MLP プロジェクタ | 2 層 MLP + GELU、ViT 出力 → LLM 埋め込み空間（LLaVA 流） | [[entities/internvl-1-5]] |
| GPT-4V | OpenAI の GPT-4 with Vision（2023 秋、商用 MLLM の代表） | [[sources/internvl-1-5]] |
| Gemini Ultra/Pro 1.0/1.5 | Google の Gemini 系列 MLLM | [[sources/internvl-1-5]] |
| Claude-3 Opus/Sonnet/Haiku | Anthropic の Claude-3 ファミリー MLLM | [[sources/internvl-1-5]] |
| Qwen-VL-Max/Plus | Alibaba の Qwen-VL 商用版 | [[sources/internvl-1-5]] |
| Grok-1.5V | xAI の MLLM、RealWorldQA を公開 | [[sources/internvl-1-5]] |
| MM1 | Apple の 30B MLLM（2024 春） | [[sources/internvl-1-5]] |
| Step-1V | StepFun の 100B MLLM | [[sources/internvl-1-5]] |
| HPT Pro | HyperGAI の MLLM | [[sources/internvl-1-5]] |
| LLaVA-NeXT | LLaVA の高解像度対応後継（2024 早春） | [[sources/internvl-1-5]] |
| Mini-Gemini | 35B MLLM（2024） | [[sources/internvl-1-5]] |
| DocOwl-1.5 | 8B 文書 MLLM | [[sources/internvl-1-5]] |
| Text-Monkey | 10B OCR MLLM | [[sources/internvl-1-5]] |
| CogVLM | 智谱 AI の VLM | [[sources/internvl-1-5]] |
| DeepSeek-VL | DeepSeek 系列（SigLIP-L + SAM-B dual encoder） | [[sources/internvl-1-5]] |
| LLaVA-HR | LLaVA High-Res（CLIP-ViT + CLIP-ConvNext dual encoder） | [[sources/internvl-1-5]] |
| mixture-of-features | Tong et al. が提案した CLIP + DINOv2 結合 | [[sources/internvl-1-5]] |
| UReader | タイル分割の祖（Ye et al., 2023） | [[sources/internvl-1-5]] |
| PaddleOCR | Baidu の OCR エンジン、InternVL 1.5 の OCR データ生成に使用 | [[sources/internvl-1-5]] |
| DocVQA | 文書理解 VQA ベンチマーク | [[sources/internvl-1-5]] |
| ChartQA | 図表理解 VQA ベンチマーク | [[sources/internvl-1-5]] |
| InfoVQA / InfographicVQA | インフォグラフィック VQA | [[sources/internvl-1-5]] |
| OCRBench | OCR 総合評価ベンチマーク | [[sources/internvl-1-5]] |
| MMBench-EN/CN | マルチモーダル総合ベンチマーク（英/中） | [[sources/internvl-1-5]] |
| CCBench | Chinese Culture Benchmark（中国文化理解） | [[sources/internvl-1-5]] |
| MMMU | Massive Multi-discipline Multimodal Understanding（多分野理解） | [[sources/internvl-1-5]] |
| AI2D | Allen Institute 2D（科学図解理解） | [[sources/internvl-1-5]] |
| MathVista | 数学的視覚推論ベンチマーク | [[sources/internvl-1-5]] |
| MMVet | 視覚汎用能力ベンチマーク | [[sources/internvl-1-5]] |
| SEED | 視覚生成・理解の総合評価ベンチマーク | [[sources/internvl-1-5]] |
| RealWorldQA / RWQA | 実世界の空間理解ベンチマーク（Grok チーム公開） | [[sources/internvl-1-5]] |
| HallusionBench / HallB | 視覚ハルシネーション評価 | [[sources/internvl-1-5]] |
| ConvBench | マルチターン対話評価（Liu et al., 2024） | [[sources/internvl-1-5]] |
| MMT-Bench | 162 subtask の包括ベンチマーク | [[sources/internvl-1-5]] |
| VLMEvalKit | InternVL 1.5 評価用フレームワーク（OpenCompass 系） | [[sources/internvl-1-5]] |
| OpenCompass | Shanghai AI Lab の LLM 評価フレームワーク | [[sources/internvl-1-5]] |
| Wukong-OCR | Wukong（華為の中国語データ）+ PaddleOCR で生成 | [[sources/internvl-1-5]] |
| LaionCOCO-OCR | LaionCOCO + PaddleOCR で生成 | [[sources/internvl-1-5]] |
| GRIT | Grounded Image-Text dataset（KOSMOS-2 系） | [[sources/internvl-1-5]] |
| All-Seeing | Wang et al. 2023 の universal perception データ | [[sources/internvl-1-5]] |
| SynthDoG | Synthetic Document Generator（合成文書 OCR） | [[sources/internvl-1-5]] |
| ALLaVA | バイリンガル LLaVA 指示データ | [[sources/internvl-1-5]] |
| ShareGPT4V | GPT-4V で生成したバイリンガル指示データ | [[sources/internvl-1-5]] |
| LVIS-Instruct4V | LVIS 指示データ | [[sources/internvl-1-5]] |
| OpenHermes2.5 | テキスト指示データ | [[sources/internvl-1-5]] |
| COIG-CQIA | 中国語指示データ | [[sources/internvl-1-5]] |
| Nous-Hermes-2-Yi-34B | Yi-34B ベースのチャット LLM（InternVL 1.2 が使用） | [[sources/internvl-1-5]] |
| Mini-InternVL | OpenGVLab の軽量 MLLM 系列（2024 Oct） | [[sources/mini-internvl]] / [[entities/mini-internvl]] |
| Mini-InternVL-1B / 2B / 4B | Mini-InternVL の 3 サイズ（Qwen2-0.5B / InternLM2-1.8B / Phi-3-Mini ベース） | [[entities/mini-internvl]] |
| Mini-InternVL-DA | Domain-Adapted Mini-InternVL（自律走行/医療/リモセン特化版） | [[entities/mini-internvl]] |
| InternViT-300M | CLIP-ViT-L 初期化 + InternViT-6B 蒸留の 300M VFM | [[entities/internvit-300m]] |
| Pixel Unshuffle | InternVL 1.5 の Pixel Shuffle と同じ（呼称違い）、4 patches → 1 token | [[sources/mini-internvl]] |
| Qwen2-0.5B | Alibaba の超軽量 LLM | [[entities/mini-internvl]] |
| InternLM2-1.8B | Shanghai AI Lab の超軽量 LLM | [[entities/mini-internvl]] |
| Phi-3-Mini | Microsoft の軽量 LLM（3.8B） | [[entities/mini-internvl]] |
| InternVL2 / InternVL2-Llama3-76B | InternVL シリーズ第 5 世代（2024 Jul）、1B-108B レンジ | [[sources/mini-internvl]] |
| InternVL4Drive-v2 | InternVL 1.5 派生の 26B 自律走行 SOTA（CVPR 2024 Challenge） | [[sources/mini-internvl]] |
| DriveLM / DriveLM-nuScenes | 自律走行 VLM ベンチマーク（317K サンプル、CVPR 2024 Challenge） | [[sources/mini-internvl]] |
| DriveGPT4 / DriveMLM / DriveVLM | 自律走行 VLM 系列 | [[sources/mini-internvl]] |
| BDD-X | Berkeley Deep Drive eXplanation（DriveGPT4 が使用） | [[sources/mini-internvl]] |
| MME-Realworld (AD) | MME の自律走行サブドメイン | [[sources/mini-internvl]] |
| CAM_FRONT / CAM_BACK / CAM_LEFT 等 | nuScenes の 6 視点カメラ位置 | [[sources/mini-internvl]] |
| Action Description / Justification | 自律走行 VLM の動作記述と理由付けタスク | [[sources/mini-internvl]] |
| Speed Signal / Turning Angle | 自律走行 VLM の制御信号予測タスク | [[sources/mini-internvl]] |
| ADAPT | 自律走行説明モデル（Mini-InternVL の比較相手） | [[sources/mini-internvl]] |
| GMAI-MMBench | 医療 AI 総合 MLLM ベンチマーク | [[sources/mini-internvl]] |
| PMC-OA / PMC-VQA / PMC-Image | PubMed Central の医療画像-テキスト対 | [[sources/mini-internvl]] |
| MedICaT / MedPix | 医療画像データセット | [[sources/mini-internvl]] |
| MIMIC-CXR | MIT の胸部 X 線データ | [[sources/mini-internvl]] |
| Open-i | NIH の医療画像データ | [[sources/mini-internvl]] |
| Quilt-1M | 組織病理学画像-テキスト対 | [[sources/mini-internvl]] |
| RP3D | X 線画像-テキスト対 | [[sources/mini-internvl]] |
| Retina Image Bank | 網膜画像データ | [[sources/mini-internvl]] |
| LLaVA-Med / Qilin-Med-VL / RadFM | 医療特化 MLLM | [[sources/mini-internvl]] |
| 2D Seg C / 2D Seg M / 2D Cls / 2D Det / 2D Mcls | GMAI-MMBench の医療評価指標 | [[sources/mini-internvl]] |
| GeoChat | リモートセンシング MLLM（LLaVA-1.5 + 7B） | [[sources/mini-internvl]] |
| EarthGPT / SkyEyeGPT / SkySenseGPT | リモートセンシング MLLM 系列 | [[sources/mini-internvl]] |
| RSVQA / RSVQA-LR / RSVQA-HR | リモートセンシング VQA ベンチマーク（Low/High Res） | [[sources/mini-internvl]] |
| FIT-RS | リモートセンシング VQA データ | [[sources/mini-internvl]] |
| DIOR-RSVG | リモートセンシング視覚的グラウンディングデータ | [[sources/mini-internvl]] |
| ChemVLM | 化学特化 MLLM | [[sources/mini-internvl]] |
| Cambrian-1 | NYU の 7B MLLM（Mini-InternVL の比較相手） | [[sources/mini-internvl]] |
| Qwen2-VL / Qwen2-VL-2B / Qwen-VL-Chat | Alibaba の MLLM 系列 | [[sources/mini-internvl]] |
| LLaVA-OneVision | LLaVA の動画拡張版（2024 Aug） | [[sources/mini-internvl]] |
| LLaVA-NeXT-mistral-7B | LLaVA-NeXT の mistral 版 | [[sources/mini-internvl]] |
| MiniCPM-V 2.0 | Tsinghua の軽量 MLLM（3B） | [[sources/mini-internvl]] |
| DeepSeek-VL-1.3B | DeepSeek の軽量 MLLM | [[sources/mini-internvl]] |
| Fuyu / MoMa / Chameleon | 視覚エンコーダ不要の MLLM 系 | [[sources/mini-internvl]] |
| ZeRO1 | DeepSpeed のメモリ最適化技法（Zero Redundancy Optimizer Stage 1） | [[sources/mini-internvl]] |
| Negative Cosine Similarity Loss | InternViT-300M の蒸留損失 | [[sources/mini-internvl]] |
| Domain Adaptation (DA) | Mini-InternVL の独自フレームワーク、特定ドメインへの転移 | [[sources/mini-internvl]] |
| VQA Format | Mini-InternVL のドメイン適応で全タスクを定式化する統一形式 | [[sources/mini-internvl]] |
| ref / box / image / IMG_CONTEXT | InternVL 系の特殊トークン | [[sources/mini-internvl]] |
| Multi-view Images | 6 視点（自律走行）の画像処理形式 | [[sources/mini-internvl]] |
| Interleaved Image Format | "Frame1: <img>..." の動画表現形式 | [[sources/mini-internvl]] |
| CVPR 2024 Autonomous Driving Challenge | DriveLM 公式リーダーボード | [[sources/mini-internvl]] |
| InternVL 2.5 | InternVL シリーズ第 6 世代（2024 Dec）、MMMU 70% 突破の初 OS MLLM | [[sources/internvl-2-5]] / [[entities/internvl-2-5]] |
| InternVL 2.0 | InternVL シリーズ第 5 世代（2024 Jul）、1B-76B、論文なし | [[sources/internvl-2-5]] |
| InternVL2-1B/2B/4B/8B/26B/40B/76B | InternVL 2.0 のサイズバリアント（vs 2.5 と比較頻出） | [[sources/internvl-2-5]] |
| InternVL2.5-1B/2B/4B/8B/26B/38B/78B | InternVL 2.5 の 7 サイズ | [[entities/internvl-2-5]] |
| InternVL2.5-Pro | InternVL 2.5 の非公開最強モデル | [[entities/internvl-2-5]] |
| InternViT-6B-448px-V2.5 | InternViT-6B 第 4 版（45 層、5.5B、dynamic 448、NTP loss） | [[entities/internvl-2-5]] |
| InternViT-300M-448px-V2.5 | InternViT-300M 第 3 版（蒸留 + V2.5 多様データで段階訓練） | [[entities/internvl-2-5]] / [[entities/internvit-300m]] |
| InternLM 2.5 | Shanghai AI Lab の最新 LLM 系列（1.8B/7B/20B-Chat） | [[entities/internvl-2-5]] |
| Qwen 2.5 | Alibaba の最新 LLM 系列（0.5B/3B/32B/72B-Instruct） | [[entities/internvl-2-5]] |
| NTP loss | Next Token Prediction loss（自己回帰 LM の標準損失、InternViT V1.0+ が採用） | [[sources/internvl-2-5]] |
| QK-Norm | Query-Key Normalization（InternViT-6B の安定化技法） | [[sources/internvl-2-5]] |
| Pixel Unshuffle / Pixel Shuffle | 空間 → チャンネル変換（4 patches → 1 token、1/4 圧縮） | [[sources/internvl-2-5]] |
| Stage 1 / 1.5 / 2 | InternVL 2.5 の 3 段階訓練（MLP warmup / ViT incremental [任意] / Full instruction tuning） | [[entities/internvl-2-5]] |
| Progressive Scaling Strategy | 小型 LLM で ViT 訓練 → 大型 LLM に転送、Stage 1.5 スキップ可（本論文で公式化） | [[sources/internvl-2-5]] |
| Test-Time Scaling | 推論時計算を増やして性能向上（CoT + Majority Voting） | [[sources/internvl-2-5]] |
| Chain-of-Thought (CoT) | "Let's think step by step" で明示的推論（MMMU で +3.7） | [[sources/internvl-2-5]] |
| Majority Voting | 複数応答の多数決でロバスト化 | [[sources/internvl-2-5]] |
| Random JPEG Compression | 品質 75-100 ランダム JPEG 圧縮 augmentation | [[sources/internvl-2-5]] |
| Square Averaging Loss | w_i = 1/x^0.5 の NTP 損失重み付け（token / sample averaging の中間） | [[sources/internvl-2-5]] |
| Multimodal Data Packing | 複数サンプルを長系列に連結、GPU 利用率向上 | [[sources/internvl-2-5]] |
| Repetition Detection | LLM ベースの繰り返しパターン検出フィルタ | [[sources/internvl-2-5]] |
| LLM-Based Quality Scoring | LLM が 0-10 でデータ品質スコアリング | [[sources/internvl-2-5]] |
| ChatML | LLM 訓練用のロール付き対話形式 | [[sources/internvl-2-5]] |
| MMMU-Pro | MMMU の強化版（standard 10 options / vision / overall） | [[sources/internvl-2-5]] |
| MATH-Vision | 競技レベル数学（3,040 問、testmini + full） | [[sources/internvl-2-5]] |
| MathVerse | 視覚数学（2,612 問、6 バージョン） | [[sources/internvl-2-5]] |
| OlympiadBench | バイリンガル五輪・高考レベル数学・物理 | [[sources/internvl-2-5]] |
| CharXiv | 科学論文 chart 理解（RQ: reasoning / DQ: descriptive） | [[sources/internvl-2-5]] |
| VCR | Visual Caption Restoration（画像内テキスト復元） | [[sources/internvl-2-5]] |
| SEED-Bench-2-Plus | text-rich 視覚タスク（charts/maps/webs、2,300 問） | [[sources/internvl-2-5]] |
| Mantis-Eval | 複数画像推論ベンチ（217 問） | [[sources/internvl-2-5]] |
| MMIU | 7 種類の複数画像関係 × 52 タスク | [[sources/internvl-2-5]] |
| MuirBench | 12 タスク × 10 種類の複数画像関係 | [[sources/internvl-2-5]] |
| BLINK | 14 タスク視覚知覚（複数画像中心） | [[sources/internvl-2-5]] |
| MIRB | 複数画像理解 4 カテゴリ | [[sources/internvl-2-5]] |
| WildVision-Bench | 500 人手キュレーション、GPT-4o 採点 | [[sources/internvl-2-5]] |
| R-Bench | 実世界画像劣化への頑健性 | [[sources/internvl-2-5]] |
| MMBench v1.1 | MMBench の精緻版 | [[sources/internvl-2-5]] |
| MMVet v2 | MMVet 拡張版（interleaved image-text 追加） | [[sources/internvl-2-5]] |
| MMHal-Bench | 96 問の hallucination 評価（GPT-4o 採点） | [[sources/internvl-2-5]] |
| CRPE | オブジェクト関係 hallucination | [[sources/internvl-2-5]] |
| MMMB | 6 言語多言語マルチモーダル（en/zh/pt/ar/tr/ru） | [[sources/internvl-2-5]] |
| Multilingual MMBench | MMBench の 6 言語版（GPT-4 翻訳） | [[sources/internvl-2-5]] |
| MTVQA | 9 言語 text-centric VQA | [[sources/internvl-2-5]] |
| Video-MME | 全スペクトル動画分析（subtitle あり/なし） | [[sources/internvl-2-5]] |
| MVBench | 20 動画タスク（perception → cognition、16 frames） | [[sources/internvl-2-5]] |
| MMBench-Video | 動画理解 + 時間推論 | [[sources/internvl-2-5]] |
| MLVU | 長動画理解（3 分 - 2 時間） | [[sources/internvl-2-5]] |
| LongVideoBench | 長フレーム入力 referring reasoning | [[sources/internvl-2-5]] |
| CG-Bench | clue-based 動画理解（1,219 動画 + 12,000 QA） | [[sources/internvl-2-5]] |
| MMLU / CMMLU / C-Eval | LLM 5-shot 知識ベンチ | [[sources/internvl-2-5]] |
| GAOKAO-Bench | 中国高考ベース LLM ベンチ | [[sources/internvl-2-5]] |
| TriviaQA / NaturalQuestions | LLM 0-shot QA | [[sources/internvl-2-5]] |
| C3 | 中国語多肢選択読解 | [[sources/internvl-2-5]] |
| RACE | 英語高校生試験読解 | [[sources/internvl-2-5]] |
| WinoGrande / HellaSwag / BBH | 常識推論 / NLI / 難タスク推論 | [[sources/internvl-2-5]] |
| GSM8K / MATH / TheoremQA | 小学・高校・STEM 数学 | [[sources/internvl-2-5]] |
| HumanEval / MBPP / MBPP-CN | コード生成（Python） | [[sources/internvl-2-5]] |
| Linear Probing / Attention Pooling Probing | 凍結特徴の線形 / attention pooling 評価 | [[sources/internvl-2-5]] |
| Head Tuning / Full Tuning | セグメンテーション凍結 backbone + head / 全層訓練 | [[sources/internvl-2-5]] |
| UperNet | セマンティックセグ標準 head | [[sources/internvl-2-5]] |
| Grounding-DINO-L / UNINEXT-H / ONE-PEACE | 視覚的グラウンディング SOTA 比較相手（InternVL 2.5 が超え） | [[sources/internvl-2-5]] |
| TextHawk2 / Ferret-v2 / CogVLM-Grounding | grounding 機能付き MLLM | [[sources/internvl-2-5]] |
| NVLM / Molmo / LLaVA-OneVision | 同時期競合 MLLM | [[sources/internvl-2-5]] |
| MiniCPM-V 2.6 / Ovis 1.6 / Phi-3.5-Vision / Aquila-VL / Cambrian / VILA-1.5 | 同時期軽量・中型 MLLM | [[sources/internvl-2-5]] |
| InternVL4Drive-v2 | InternVL 派生の 26B 自律走行 SOTA | [[sources/internvl-2-5]] / [[sources/mini-internvl]] |
| OpenAI o1 | OpenAI の reasoning-focused モデル（2024 Sep、test-time scaling の象徴） | [[sources/internvl-2-5]] |
| InternVL 3 | InternVL シリーズ第 7 世代（2025 Apr）、Native Multimodal Pre-Training、MMMU 72.2 SOTA | [[sources/internvl-3]] / [[entities/internvl-3]] |
| InternVL3-1B/2B/8B/9B/14B/38B/78B | InternVL 3 の 7 サイズ | [[entities/internvl-3]] |
| InternLM3-8B | Shanghai AI Lab の最新 8B LLM（InternLM 2.5 後継、InternVL3-9B が使用） | [[entities/internvl-3]] |
| Qwen2.5 base | Alibaba Qwen2.5 のベースモデル（Chat/Instruct ではない、InternVL 3 が起点に使用） | [[entities/internvl-3]] |
| Native Multimodal Pre-Training | テキスト + マルチモーダル共同事前学習（本論文の中核哲学） | [[sources/internvl-3]] |
| V2PE | Variable Visual Position Encoding（視覚トークンに位置インクリメント δ < 1） | [[sources/internvl-3]] |
| δ (delta) | V2PE の視覚トークン位置インクリメント（{1, 1/2, 1/4, ..., 1/256}） | [[sources/internvl-3]] |
| MPO | Mixed Preference Optimization（DPO + BCO + LM 損失） | [[sources/internvl-3]] |
| DPO | Direct Preference Optimization（Rafailov et al., 2023） | [[sources/internvl-3]] |
| BCO | Binary Classifier Optimization（個別応答の絶対品質を独立評価） | [[sources/internvl-3]] |
| VisualPRM | Visual Process Reward Model（各推論ステップに +/- スコア） | [[sources/internvl-3]] |
| VisualPRM-8B | InternVL3-8B ベースの 8B critic モデル | [[entities/internvl-3]] |
| VisualPRM400K | VisualPRM 訓練データ（MMPR v1.2 ベース、InternVL3 で拡張） | [[sources/internvl-3]] |
| MMPR v1.2 | Multimodal Preference Repository v1.2（多モーダル選好データ） | [[sources/internvl-3]] |
| Best-of-N / Bo8 | テスト時スケーリング: N=8 個の応答から最良選択 | [[sources/internvl-3]] |
| InternEVO | InternVL 3 用に拡張された ZeRO 最適化訓練フレームワーク（50-200% 高速化） | [[sources/internvl-3]] |
| Head-Parallel | 32K トークン系列対応の並列化技法 | [[sources/internvl-3]] |
| InternVL-Data | InternVL3 公開訓練データセット（HF: `OpenGVLab/InternVL-Data`） | [[entities/internvl-3]] |
| GUI Grounding | スクリーンショット上の UI 要素の位置特定 | [[sources/internvl-3]] |
| ScreenSpot / ScreenSpot-V2 | GUI grounding ベンチマーク | [[sources/internvl-3]] |
| UI-TARS-72B | ByteDance の GUI エージェント特化 MLLM | [[sources/internvl-3]] |
| Aguvis-72B | GUI 特化 MLLM（ScreenSpot SOTA） | [[sources/internvl-3]] |
| VSI-Bench | Visual-Spatial Intelligence Benchmark（3D 空間推論） | [[sources/internvl-3]] |
| MathVision | Math-Vision の別名（InternVL 3 論文表記） | [[sources/internvl-3]] |
| DynaMath | 動的数学推論ベンチマーク | [[sources/internvl-3]] |
| WeMath | 視覚数学ベンチマーク | [[sources/internvl-3]] |
| LogicVista | 論理推論ベンチマーク | [[sources/internvl-3]] |
| Claude-3.7-Sonnet | Anthropic Claude-3 系列の reasoning 強化版 | [[sources/internvl-3]] |
| Gemini-2.0-Pro / Flash | Google Gemini-2 系列 | [[sources/internvl-3]] |
| Gemini-2.5-Pro | Google Gemini-2.5 系列（最新） | [[sources/internvl-3]] |
| Qwen2.5-VL-3B/7B/32B/72B | Alibaba Qwen2.5-VL 系列（InternVL 3 の直接競合） | [[sources/internvl-3]] |
| QvQ-72B-Preview | Alibaba の reasoning-focused 視覚 MLLM | [[sources/internvl-3]] |
| Ovis2-16B / 34B | Ovis シリーズ後継 MLLM | [[sources/internvl-3]] |
| MiniCPM-o2.6 | MiniCPM-V 2.6 後継 | [[sources/internvl-3]] |
| Oryx-1.5-32B | 動画 MLLM | [[sources/internvl-3]] |
| VideoLLaMA2-72B | DAMO 動画 MLLM | [[sources/internvl-3]] |
| OmniCorpus | 大規模多モーダル interleaved コーパス（Shanghai AI Lab） | [[sources/internvl-3]] |
| Tongyi Qianwen LICENSE | Qwen2.5 のライセンス（商用制約あり、InternVL 3 ライセンス継承） | [[sources/internvl-3]] |
| GLM-4v-Plus | 智谱 AI の MLLM（InternVL 3 比較相手） | [[sources/internvl-3]] |
| Step-1o | StepFun の MLLM（InternVL 3 比較相手） | [[sources/internvl-3]] |
| ChatGPT-4o-latest | OpenAI の最新 GPT-4o（2024 Nov 等） | [[sources/internvl-3]] |
| MPO (paper) | Mixed Preference Optimization 提案論文（Wang et al., 2024 Nov） | [[sources/mpo]] / [[entities/mpo]] |
| MMPR | MultiModal PReference dataset（約 3M ペア、本論文で公開） | [[entities/mmpr]] |
| MMPR v1 | MMPR 初版（InternVL2-8B-MPO 用） | [[entities/mmpr]] |
| MMPR v1.2 | MMPR 拡張版（InternVL 3 + VisualPRM 用） | [[entities/mmpr]] |
| DropoutNTP | Dropout Next-Token Prediction（画像なしで応答補完、本論文の独自貢献） | [[sources/mpo]] |
| InternVL2-8B-MPO | MPO 訓練後の InternVL2-8B（本論文の代表モデル） | [[entities/mpo]] |
| BCO | Binary Classifier Optimization（Jung et al. 2024、chosen→1/rejected→0） | [[sources/mpo]] |
| RLAIF-V | divide-and-conquer 方式のマルチモーダル選好データ生成（Yu et al. 2024） | [[sources/mpo]] |
| Teacher Forcing | SFT で正解トークンを与えて訓練、分布シフトの原因 | [[sources/mpo]] |
| Distribution Shift | SFT 訓練と推論のあいだの分布のズレ、CoT で深刻化 | [[sources/mpo]] |
| Bradley-Terry model | 1952 年提案、ペア比較から選好順序を推定する確率モデル（DPO 系の基礎） | [[sources/mpo]] |
| Reference Model π₀ | DPO 系で凍結される元モデル（KL 制約用） | [[sources/mpo]] |
| Policy Model π_θ | DPO 系で更新される学習可能モデル | [[sources/mpo]] |
| KL Penalty β | DPO の reference からの逸脱罰則係数（MPO では 0.1） | [[sources/mpo]] |
| Reward Shift δ | BCO の安定化用、過去報酬の moving average | [[sources/mpo]] |
| RSO | Statistical Rejection Sampling Optimization（DPO の hinge 損失版） | [[sources/mpo]] |
| IPO | Identity Preference Optimization（DPO の overfitting 対策版） | [[sources/mpo]] |
| cDPO / RobustDPO | preference label のノイズに頑健な DPO 変種 | [[sources/mpo]] |
| SPPO | Self-Play Preference Optimization | [[sources/mpo]] |
| AOT | Distributional Preference Alignment via Optimal Transport | [[sources/mpo]] |
| TR-DPO | reference model を定期的に同期する DPO 変種 | [[sources/mpo]] |
| ORPO | Odds Ratio Preference Optimization（reference model 不要） | [[sources/mpo]] |
| Smaug | Pal et al. 2024、DPO が gibberish を生む現象を分析 | [[sources/mpo]] |
| Object HalBench | object-level hallucination ベンチマーク | [[sources/mpo]] |
| InstructGPT | OpenAI の RLHF 初期実装（Ouyang et al. 2022） | [[sources/mpo]] |
| PPO-Max | PPO の安定化版 | [[sources/mpo]] |
| M3CoT | Multi-domain Multi-step Multi-modal CoT（マルチモーダル CoT ベンチ） | [[sources/mpo]] |
| Geo170K / GeoQA+ / GEOS / GeomVerse / Geometry3K / CLEVR-Math | 数学 grounding データセット（MMPR 構築用） | [[entities/mmpr]] |
| DVQA / MapQA / ChartQA | チャート VQA（MMPR） | [[entities/mmpr]] |
| OCRVQA / STVQA / SROIE / InfoVQA / TextVQA | OCR VQA（MMPR） | [[entities/mmpr]] |
| AI2D / ScienceQA | 科学 VQA（MMPR） | [[entities/mmpr]] |
| TheoremQA | 複雑科学問題（MPO で +5.2、テキスト専用） | [[sources/mpo]] |
| IFEval | Instruction-Following Evaluation（MPO で +4.1） | [[sources/mpo]] |
| GAOKAO | 中国高考、テキスト LLM 評価 | [[sources/mpo]] |
| 3H | helpful, honest, harmless（InstructGPT の目標） | [[sources/mpo]] |
| Background Knowledge-based CoT | 背景知識ベース CoT（Science 用、MMPR） | [[sources/mpo]] |
| Visual Content-based CoT | 視覚内容ベース CoT（Chart/OCR/Document 用、MMPR） | [[sources/mpo]] |
| Grounded CoT | grounding 付き CoT（General VQA 用、MMPR） | [[sources/mpo]] |
| Dropout Ratio (DR) | DropoutNTP の応答切り詰め比率（最適 0.5） | [[sources/mpo]] |
| token cost per pair | 選好ペアあたりトークン数（DropoutNTP 571.2 vs RLAIF-V 992.7） | [[sources/mpo]] |
| InternVL 3.5 | InternVL シリーズ第 8 世代（2025 Aug）、Cascade RL + MoE + ViR + DvD、MMMU 77.7 | [[sources/internvl-3-5]] / [[entities/internvl-3-5]] |
| InternVL3.5-1B/2B/4B/8B/14B/38B | InternVL 3.5 の Dense モデル 6 サイズ | [[entities/internvl-3-5]] |
| InternVL3.5-20B-A4B | InternVL 3.5 の MoE モデル（GPT-OSS-20B、activated 4B） | [[entities/internvl-3-5]] |
| InternVL3.5-30B-A3B | InternVL 3.5 の MoE モデル（Qwen3-30B-A3B、activated 3B） | [[entities/internvl-3-5]] |
| InternVL3.5-241B-A28B | InternVL 3.5 の最大 MoE モデル（Qwen3-235B-A22B、activated 28B） | [[entities/internvl-3-5]] |
| InternVL3.5-Flash | ViR を統合した InternVL 3.5 効率版（視覚トークン 50% 削減で性能 99% 維持） | [[entities/internvl-3-5]] |
| Cascade RL | Cascade Reinforcement Learning（offline MPO + online GSPO の 2 段階、InternVL 3.5 の中核） | [[sources/internvl-3-5]] |
| GSPO | Geometric mean Sequence-level PPO（online RL、reference 制約なし、トークン単位 geometric mean ratio） | [[sources/internvl-3-5]] |
| GRPO | Group Relative Policy Optimization（DeepSeek 系の online RL、GSPO の元） | [[sources/internvl-3-5]] |
| GSPO importance ratio | $s_i = (\pi_\theta/\pi_{\text{old}})^{1/\|y_i\|}$、トークン単位の geometric mean | [[sources/internvl-3-5]] |
| ViR | Visual Resolution Router（patch ごとに 1/4 or 1/16 圧縮率を動的選択） | [[sources/internvl-3-5]] |
| ViCO | Visual Consistency Learning（ViR 訓練用 2 段階手法: consistency + router training） | [[sources/internvl-3-5]] |
| patch router | ViR のバイナリ分類器（cross-entropy 訓練） | [[sources/internvl-3-5]] |
| loss ratio r_i | ViR target 用、L(I_1/16) / L(I_1/4) | [[sources/internvl-3-5]] |
| k-th percentile threshold τ | ViR の動的閾値（target 分布をバランス） | [[sources/internvl-3-5]] |
| Pixel Shuffle (1/16) | InternVL 3.5 の高圧縮率版（patch を 64 token に圧縮） | [[sources/internvl-3-5]] |
| DvD | Decoupled Vision-Language Deployment（ViT/MLP を vision server、LLM を language server に分離） | [[sources/internvl-3-5]] |
| Vision Server | DvD の視覚処理サーバ（ViT + MLP + ViR） | [[sources/internvl-3-5]] |
| Language Server | DvD の言語処理サーバ（LLM のみ） | [[sources/internvl-3-5]] |
| 非同期 3 段階パイプライン | DvD の Vision → Transfer → Language の重ね合わせ実行 | [[sources/internvl-3-5]] |
| RDMA | Remote Direct Memory Access（DvD 視覚特徴転送の高速オプション） | [[sources/internvl-3-5]] |
| Qwen3 series | Alibaba の最新 LLM（0.6B/1.7B/4B/8B/14B/32B/30B-A3B/235B-A22B、InternVL 3.5 の主軸） | [[entities/internvl-3-5]] |
| GPT-OSS-20B | OpenAI 公開の MoE LLM（InternVL 3.5-20B-A4B が採用） | [[entities/internvl-3-5]] |
| Activated Parameters (A4B/A3B/A28B) | MoE モデルの活性パラメータ数表記 | [[entities/internvl-3-5]] |
| Thinking Mode | Qwen3 等の reasoning モード（Deep Thinking に使用） | [[sources/internvl-3-5]] |
| Deep Thinking | InternVL 3.5 の Test-Time Scaling 手法（Thinking モード起動の step-by-step 推論） | [[sources/internvl-3-5]] |
| Parallel Thinking | InternVL 3.5 の Test-Time Scaling 手法（Best-of-N + VisualPRM-v1.1） | [[sources/internvl-3-5]] |
| VisualPRM-v1.1 | InternVL 3.5 の critic モデル（InternVL 3 の VisualPRM 後継） | [[entities/internvl-3-5]] |
| MMPR-Tiny | MMPR-v1.2 から accuracy 0.2-0.8 でフィルタリングした online RL データ（70K クエリ） | [[entities/mmpr]] / [[entities/internvl-3-5]] |
| InternEVO / XTuner | InternVL 3.5 の訓練フレームワーク | [[sources/internvl-3-5]] |
| FSDP | Fully Shared Data Parallelism（XTuner の最適化） | [[sources/internvl-3-5]] |
| DeepGEMM | FP8 GEMM カーネル（DeepSeek 提案、XTuner で使用） | [[sources/internvl-3-5]] |
| liger-kernel | fused cross-entropy operator | [[sources/internvl-3-5]] |
| FlashAttention-3 | 高速 attention 実装（packed inputs 対応） | [[sources/internvl-3-5]] |
| TMA-Adaptive FP8 Grouped GEMM | MoE 訓練用の特殊カーネル | [[sources/internvl-3-5]] |
| verl | online RL 段階のコードベース | [[sources/internvl-3-5]] |
| window attention with sink | GPT-OSS-20B の特殊 attention（InternVL3.5-20B-A4B で Triton 版実装） | [[sources/internvl-3-5]] |
| GPT-5 / GPT-5-nano | OpenAI の最新フロンティア MLLM（2025 Aug） | [[sources/internvl-3-5]] |
| GLM-4.1V-9B / GLM-4.5V | 智谱 AI の最新マルチモーダル MLLM | [[sources/internvl-3-5]] |
| Step-3 / Step3-321B-A38B | StepFun の MLLM | [[sources/internvl-3-5]] |
| Kimi-VL-A3B-2506 | Moonshot Kimi-VL の MoE 版 | [[sources/internvl-3-5]] |
| MiMo-VL-RL-8B | Xiaomi の RL 強化 MLLM | [[sources/internvl-3-5]] |
| Keye-VL-8B | 快手 Keye の MLLM | [[sources/internvl-3-5]] |
| Ovis-2B/4B/8B/2-16B/2-34B | Ovis シリーズ MLLM | [[sources/internvl-3-5]] |
| MiniCPM-V-4 / MiniCPM-o-2.6 | Tsinghua の軽量 MLLM | [[sources/internvl-3-5]] |
| Skywork-R1V3-38B | 昆仑万维 Skywork の reasoning MLLM | [[sources/internvl-3-5]] |
| QvQ-72B-Preview | Alibaba の reasoning MLLM | [[sources/internvl-3-5]] |
| Doubao-1.5-Pro | ByteDance Doubao | [[sources/internvl-3-5]] |
| Seed1.5-VL | ByteDance の MLLM（ScreenSpot-v2 商用 SOTA） | [[sources/internvl-3-5]] |
| WindowsAgentArena | Windows GUI エージェントベンチ | [[sources/internvl-3-5]] |
| WebArena-Lite-v2 | Web エージェントベンチ | [[sources/internvl-3-5]] |
| OSWorld / OSWorld-G | OS-level GUI エージェントベンチ | [[sources/internvl-3-5]] |
| SGP-Bench | SVG 理解ベンチ | [[sources/internvl-3-5]] |
| SArena-Icon | SVG 生成ベンチ（Text2SVG / Img2SVG） | [[sources/internvl-3-5]] |
| ERQA | embodied reasoning ベンチ | [[sources/internvl-3-5]] |
| SpaCE-10 | 空間理解ベンチ | [[sources/internvl-3-5]] |
| OmniSpatial | 包括的空間ベンチ | [[sources/internvl-3-5]] |
| Llama-4-Scout / Maverick | Meta Llama-4 系 | [[sources/internvl-3-5]] |
| DeepSeek-V3-671B-A37B | DeepSeek の最大 MoE | [[sources/internvl-3-5]] |
| Reward Hacking | RL モデルが reward を gaming する現象 | [[sources/internvl-3-5]] |
| Request Throughput | DvD の評価指標（requests/s） | [[sources/internvl-3-5]] |
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
| RoPE | Rotary Position Embeddings（回転位置埋め込み） | [[concepts/rotary-position-embeddings]] / [[sources/roformer]] |
| YaRN | Yet another RoPE extensioN method（RoPE 拡張系の決定的手法、NTK-by-parts + attention 温度スケーリング） | [[sources/yarn]] / [[concepts/rotary-position-embeddings]] |
| DFN | Data Filtering Networks（データ・フィルタリング・ネットワーク、Apple 2023） | [[sources/dfn]] / [[entities/dfn]] |
| DFN-2B / DFN-5B | DFN が DataComp / 42B プールから誘導したデータセット | [[entities/dfn]] |
| HQITP-350M / HQITP-135M | High-Quality Image-Text Pairs（Apple 内部の人間検証済みキャプション 357M / 135M） | [[entities/dfn]] |
| DataComp | データセット評価ベンチマーク（Gadre et al. 2023、12.8B 画像-テキスト対 + 38 タスク） | [[sources/dfn]] |
| DataComp-1B (DC-1B) | DataComp 提供のベースライン・データセット（CLIP フィルタ + IN クラスタリング） | [[sources/dfn]] |
| CommonPool | DataComp の未フィルタ 12.8B プール | [[sources/dfn]] |
| filter dataset | DFN 訓練に使うデータプール | [[sources/dfn]] |
| induced dataset | DFN がフィルタリングして生成したデータセット | [[sources/dfn]] |
| induced model | 誘導データセットで訓練したモデル | [[sources/dfn]] |
| OAI-Init | OpenAI CLIP 重みでの初期化 | [[sources/dfn]] |
| M3AE | Multimodal Masked Autoencoder（DFN の代替フィルタ候補） | [[sources/dfn]] |
| AXlearn | Apple の TPU 訓練フレームワーク | [[sources/dfn]] |
| CC12M / CC3M | Conceptual Captions 12M / 3M（DFN 公開データ訓練） | [[sources/dfn]] |
| SS15M | Shutterstock 15M（DFN 公開データ訓練） | [[sources/dfn]] |
| PI | Position Interpolation（位置補間、Chen 2023 / Kaiokendev 2023） | [[sources/yarn]] |
| NTK-aware (RoPE) | Neural Tangent Kernel aware 補間、周波数を基底変換 $b'$ で再分配 | [[sources/yarn]] |
| NTK-by-parts | 波長 $\lambda_d$ ごとに補間戦略を変えるターゲット補間 | [[sources/yarn]] |
| Dynamic NTK / Dynamic Scaling | 推論時に scale factor $s$ を動的更新する技法 | [[sources/yarn]] |
| Dynamic-YaRN | YaRN を Dynamic Scaling と組み合わせた推論時手法 | [[sources/yarn]] |
| LongRoPE | 進化アルゴリズムで非一様補間係数を探索、200K+ 系列対応 | [[concepts/rotary-position-embeddings]] |
| scale factor $s$ | 拡張比率 $L'/L$（PI 以降の全 RoPE 拡張手法で共通） | [[sources/yarn]] |
| wavelength $\lambda_d$ | $2\pi b^{2d/\|D\|}$、次元 $d$ で完全な回転を行うのに必要なトークン長 | [[sources/yarn]] |
| ramp function $\gamma(r)$ | NTK-by-parts の境界遷移関数、$\alpha, \beta$ で制御 | [[sources/yarn]] |
| Passkey Retrieval | 大量無意味テキスト中の 5 桁数字検索タスク（Mohtashami & Jaggi 2023） | [[sources/yarn]] |
| PG19 | Project Gutenberg books の長文書データセット（YaRN 訓練用） | [[sources/yarn]] |
| Proof-pile | 数学証明の長文書評価データセット | [[sources/yarn]] |
| GovReport | 政府報告書の長文書データセット | [[sources/yarn]] |
| Flash Attention 2 | YaRN と直接互換な高速 attention 実装（Dao 2023） | [[sources/yarn]] |
| bloc97 | Reddit ハンドル、NTK-aware / NTK-by-parts 発見者 = YaRN 著者 Bowen Peng | [[sources/yarn]] |
| emozilla | Reddit ハンドル、Dynamic NTK 発見者 = YaRN 著者 Jeffrey Quesnelle | [[sources/yarn]] |
| ViT | Vision Transformer（画像を 16×16 パッチに切り Transformer に流す画像認識モデル） | [[concepts/vision-transformer]] / [[sources/vision-transformer]] |
| ViT-B/L/H | ViT のサイズ命名規約: Base(86M)/Large(307M)/Huge(632M) | [[sources/vision-transformer]] |
| MSA | Multi-head Self-Attention（多頭自己注意） | [[sources/vision-transformer]] / [[concepts/vision-transformer]] |
| JFT-300M | Google 内部の非公開 3 億画像データセット | [[sources/vision-transformer]] |
| BiT | Big Transfer（Kolesnikov et al., 2020、ResNet ベースの大規模転移学習） | [[sources/vision-transformer]] |
| Noisy Student | EfficientNet ベース半教師あり SOTA（Xie et al., 2020） | [[sources/vision-transformer]] |
| VTAB | Visual Task Adaptation Benchmark（19 タスク、Natural/Specialized/Structured 3 分類） | [[sources/vision-transformer]] |
| iGPT | image GPT（Chen et al., 2020、ピクセルを直接 Transformer に） | [[sources/vision-transformer]] |
| TPUv3-core-days | TPU v3 コア数 × 訓練日数（事前学習計算コストの単位） | [[sources/vision-transformer]] |
| attention distance | attention 重みから計算される画像空間での情報統合の平均距離（CNN の受容野類比） | [[sources/vision-transformer]] |
| inductive bias | 帰納バイアス（モデル構造に組み込まれた仮定、例: CNN の局所性・平行移動同変性） | [[sources/vision-transformer]] |
| GELU | Gaussian Error Linear Unit（ViT の MLP で採用される活性化関数） | [[sources/vision-transformer]] |
| [CLS] token | classification token（BERT 由来、ViT で画像全体を集約する学習可能トークン） | [[sources/vision-transformer]] / [[concepts/vision-transformer]] |
| pre-norm | LayerNorm を残差ブロックの前に置く Transformer 構造（ViT で標準採用） | [[sources/vision-transformer]] |
| RoFormer | Rotary Former（RoPE を組み込んだ Transformer の名称、原典論文のモデル名） | [[sources/roformer]] |
| PLM | Pre-trained Language Model（事前学習済み言語モデル） | [[sources/roformer]] |
| GLUE | General Language Understanding Evaluation（NLP 標準ベンチマーク集） | [[sources/roformer]] |
| MRPC / SST-2 / QNLI / STS-B / QQP / MNLI | GLUE 内の個別タスク | [[sources/roformer]] |
| CAIL2019-SCM | Chinese AI and Law 2019 Similar Case Matching（中国法律文書類似マッチング） | [[sources/roformer]] |
| WoBERT | Word-based BERT for Chinese（中国語向け単語ベース BERT、RoFormer 著者 Su Jianlin の別作品） | [[sources/roformer]] |
| NEZHA | 華為製の中国語事前学習済み Transformer | [[sources/roformer]] |
| Performer | 線形 attention を実装した Transformer 変種（Choromanski et al., 2020） | [[sources/roformer]] |
| long-term decay | 長期減衰（相対距離が大きくなると attention 寄与が単調減衰する性質） | [[sources/roformer]] |
| Abel transformation | アーベル変換（RoPE の長期減衰証明で使われる級数操作） | [[sources/roformer]] |
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
