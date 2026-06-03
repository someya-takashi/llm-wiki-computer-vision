---
type: source
source_path: raw/papers/InternVL3_ Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models.md
source_kind: paper
title: "InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models"
authors: [Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, Zhangwei Gao, Erfei Cui, Xuehui Wang, Yue Cao, Yangzhou Liu, Xingguang Wei, Hongjie Zhang, Haomin Wang, Weiye Xu, Hao Li, Jiahao Wang, Nianchen Deng, Songze Li, Yinan He, Tan Jiang, Jiapeng Luo, Yi Wang, Conghui He, Botian Shi, Xingcheng Zhang, Wenqi Shao, Junjun He, Yingtong Xiong, Wenwen Qu, Peng Sun, Penglong Jiao, Han Lv, Lijun Wu, Kaipeng Zhang, Huipeng Deng, Jiaye Ge, Kai Chen, Limin Wang, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, Wenhai Wang]
year: 2025
venue: "arXiv:2504.10479 (technical report)"
ingested: 2026-05-29
tags: [mllm, vllm, internvl-3, native-multimodal-pretraining, v2pe, mpo, mixed-preference-optimization, visualprm, mmmu-72-2, opengvlab]
translation: "[[translations/internvl-3]]"
---

# InternVL 3 — Native Multimodal Pre-Training パラダイムを確立、MMMU 72.2 で新 SOTA

> 原典: [[translations/internvl-3]] ・ `raw/papers/InternVL3_ Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models.md`
> 著者: Jinguo Zhu, Weiyun Wang, Zhe Chen ら（Shanghai AI Lab + SenseTime + 清華大学 + 南京大学 + 復旦大学 + 香港中文大学 + Shanghai Jiao Tong University）
> 投稿: arXiv:2504.10479（2025 年 4 月、technical report）
> リポジトリ: <https://github.com/OpenGVLab/InternVL>
> モデル: <https://huggingface.co/OpenGVLab/InternVL3-78B>
> データ: <https://huggingface.co/datasets/OpenGVLab/InternVL-Data>

---

## 一言まとめ

**「[[entities/internvl-2-5|InternVL 2.5]] の「アーキテクチャは変えず、訓練・データ・テスト時で勝つ」哲学をさらに推し進め、**訓練パラダイム自体を「事後的 MLLM 適応」から「Native Multimodal Pre-Training」へ転換** した論文」**。LLM Chat 版から MLLM を「事後改造」する従来手法（InternVL 1.0-2.5 含む）と異なり、**InternVL 3 は最初から Qwen2.5 base + InternViT で、テキスト + マルチモーダルデータを共同で事前学習** する。さらに **Variable Visual Position Encoding (V2PE) で長文脈対応、Mixed Preference Optimization (MPO) で CoT 推論強化、VisualPRM で test-time scaling**。**MMMU 72.2 で再びオープンソース SOTA**、**1B/2B/8B/9B/14B/38B/78B の 7 サイズ + InternLM3-8B バリアント**。**Qwen2.5 Chat と同じ base から派生して言語能力が向上**（オープン MLLM が「マルチモーダル化で純粋言語性能を上げた」初の本格的実証）。**訓練データとモデル重みを完全公開**（"open-science" 原則を強調）、Apache 2.0 + Qwen ライセンス。

---

## 背景と問題意識

### 「事後的（post-hoc）MLLM 訓練」への根本的疑念

[[sources/internvl|InternVL 1.0]] から [[sources/internvl-2-5|InternVL 2.5]] までのシリーズは、すべて **「LLM Chat 版を base に事後的にマルチモーダル能力を付与する」** という伝統的パラダイムに従ってきた:

```
[従来パラダイム（InternVL 1.0 - 2.5、LLaVA / Qwen2-VL も同じ）]
   ステップ 1: テキスト専用 LLM 事前学習（base model）
        ↓
   ステップ 2: 純粋言語の post-training（→ Chat model）
        ↓
   ステップ 3: MLLM 適応（MLP warmup → Full instruction tuning）
        - 補助データ（OCR 等）が必要
        - パラメータ凍結や多段階微調整スケジュールが必要
        - 純粋言語能力が低下しやすい（InternVL 2.0 で -2.1〜-2.3）
```

著者らはこのパラダイムに **根本的疑念** を呈する: 「**なぜ言語専用に最適化された Chat モデルから始めなければいけないのか？最初から両モダリティを共同で学習すれば、整列問題は本質的に解消するのでは？**」

### この論文の位置づけ

**「InternVL シリーズの哲学転換」** を象徴する論文。

| シリーズ | アーキテクチャ | 訓練パラダイム | 主役技術 |
|---|---|---|---|
| InternVL 1.0 | InternViT-6B + QLLaMA + Vicuna | 3 段階（contrastive + ITC/ITM/ITG + SFT） | QLLaMA |
| InternVL 1.5 / 2.0 / 2.5 | ViT-MLP-LLM（Chat 版 LLM 起点） | 事後的（MLP warmup → ViT incremental → SFT） | 動的解像度 + データ |
| **InternVL 3** | ViT-MLP-LLM（**base LLM 起点**） | **Native Multimodal Pre-Training**（言語 + マルチモーダル共同事前学習） | **V2PE + MPO** |

**「商用追従ライン最終到達点」** だった [[entities/internvl-2-5|InternVL 2.5]] から **「商用超え + 新パラダイム確立」** へ。

---

## 提案手法

### 1. Native Multimodal Pre-Training（本論文の最大貢献）

**最も重要な変更**: LLM コンポーネントは **base モデル**（Qwen2.5-0.5B/1.5B/7B/14B/32B/72B、InternLM3-8B）から初期化。**Chat / Instruct 版ではない**。

```
[Native Multimodal Pre-Training（InternVL 3）]
   ステップ 1: 単一段階で同時実行
        - 純粋言語データ（InternLM2.5 由来 + オープンソース、~50B トークン）
        - マルチモーダルデータ（InternVL 2.5 由来 + GUI/Tool/3D/Video、~150B トークン）
        - 共同パラメータ最適化（ViT + MLP + LLM 全層）
        ↓
   ステップ 2 (SFT): 21.7M サンプル（2.5 の 16.3M から +5.4M）
        ↓
   ステップ 3 (MPO): 300K 選好データで CoT 推論強化
```

**サンプリング比率**: **言語:マルチモーダル = 1:3** が最適（2 段階探索で発見）。総 ~200B トークン（言語 50B + マルチモーダル 150B）。

#### 損失関数: text-only loss + square averaging

$$\mathcal{L}_{\text{text-only}}(\theta) = -\sum_{i=2, x_i \in \text{Text}}^L w_i \cdot \log p_\theta(x_i | x_1, \ldots, x_{i-1})$$

- **損失計算はテキストトークンのみ**（視覚トークンは条件付け文脈、直接予測しない）
- 重み $w_i = 1/l^{0.5}$（square averaging、[[entities/internvl-2-5|InternVL 2.5]] と同じ）

#### 共同パラメータ最適化

**「ViT + MLP + LLM の全層を共同訓練」**。従来パイプラインのような層凍結なし。これにより:

- テキスト表現と視覚特徴が **同期的に進化**
- 最終パラメータが純粋言語と多モーダル両方で **追加調整なしで高性能**

> **補足: なぜ「base LLM」なのか** — Chat 版 LLM は既に「人間の好み」に調整されており、視覚情報を組み込む際に「言語の硬直性」が干渉する。Base モデルは「素の言語能力」のみで、視覚モダリティ統合の余地が大きい。これは画家が **真っ白なキャンバスから絵を描き始める** か、**既存の絵に追加描画する** かの違いに似ている。

### 2. Variable Visual Position Encoding (V2PE)

**長いマルチモーダル文脈を扱うための位置エンコーディング革新**。

```
[従来の Position Encoding]
   p_i = p_{i-1} + 1 （テキストでも視覚でも一律 +1）

[V2PE]
   p_i = p_{i-1} + 1   （x_i がテキストトークン）
   p_i = p_{i-1} + δ   （x_i が視覚トークン、δ < 1）
```

**δ の選択**: 訓練中は $\Delta = \{1, 1/2, 1/4, ..., 1/256\}$ からランダム。**画像内では δ 一定**（相対位置保持）。推論時は系列長に応じて柔軟選択。$\delta = 1$ で従来 InternVL 2.5 に帰着。

**効果**: 短文脈タスクでも **$\delta = 1/4$ が最適**（表 12）、長文脈タスクではさらに有利。

> **補足: なぜ V2PE が効くか** — 視覚トークンは画像の隣接パッチ間に強い局所的相関があり、テキストトークンほど「位置の独立性」がない。$\delta < 1$ で **「複数視覚トークンを 1 つの『大きな位置』として扱う」** ことで、位置空間の浪費を防ぎ、長文脈をモデルの有効文脈窓に収める。

### 3. Mixed Preference Optimization (MPO)

**CoT 推論を強化する後訓練技法**。Pre-training + SFT は teacher-forcing（正解条件付き予測）だが、推論時は自分の出力に基づくため **分布シフト** が発生し CoT が壊れる。MPO で正例・負例両方から監督。

**損失**:
$$\mathcal{L} = w_p \mathcal{L}_p + w_q \mathcal{L}_q + w_g \mathcal{L}_g$$

- **$\mathcal{L}_p$（DPO loss）**: chosen と rejected の相対選好（β=KL 罰則）
- **$\mathcal{L}_q$（BCO loss）**: chosen / rejected の絶対品質を独立評価
- **$\mathcal{L}_g$（LM loss）**: 選好応答の生成プロセス

**データ**: MMPR v1.2（300K サンプル、VQA / 科学 / チャート / 数学 / OCR / 文書をカバー）。SFT 版 8B/38B/78B で rollouts 生成。

**効果**（表 13）:

| Model | w/o MPO | w/ MPO | Δ |
|---|---|---|---|
| InternVL3-1B | 24.6 | 25.1 | +0.5 |
| InternVL3-14B | 46.2 | 49.9 | **+3.7** |
| InternVL3-38B | 48.3 | 52.8 | **+4.5** |
| InternVL3-78B | 50.5 | 54.6 | **+4.1** |

**MPO データは SFT のサブセット** → 性能改善が **アルゴリズムによる** ことを実証。大型モデルほど効果大。

### 4. Test-Time Scaling: VisualPRM-8B

**Visual Process Reward Model**。各推論ステップに +/- スコアを付与、平均してソリューション全体の品質を評価。**Best-of-N 戦略** で最良応答選択。

**訓練**: VisualPRM400K（MMPR v1.2 ベース、InternVL3-8B/38B で rollouts 拡張）。
**推論**: 各ステップで「+」生成確率がスコア。

**効果（表 2）**: 小型モデルほど顕著:

| Model | w/o Bo8 | w/ Bo8 | Δ |
|---|---|---|---|
| InternVL3-1B | 25.1 | **35.0** | **+9.9** |
| InternVL3-2B | 32.4 | **41.7** | **+9.3** |
| InternVL3-8B | 44.3 | **50.2** | **+5.9** |
| InternVL3-78B | 54.6 | **56.5** | **+1.9** |

> **重要観察**: 小型モデル + Best-of-N が大型モデルに匹敵し得る。**「計算リソースをモデルサイズで使うか、推論回数で使うか」** という新しいトレードオフ。

### 5. インフラ強化: InternEVO

**LLM 大規模訓練用 ZeRO 最適化フレームワークを InternVL 用に拡張**。ViT/MLP/LLM の柔軟・分離型シャーディング、data/tensor/sequence/pipeline parallelism。**InternVL 2.5 比で 50-200% 訓練高速化**（同計算予算）。

### モデルカタログ（7 サイズ + InternLM3 バリアント）

| Model | LLM Base | LLM Provider | OpenCompass |
|---|---|---|---|
| **InternVL3-1B** | Qwen2.5-0.5B base | Alibaba | 57.4 |
| **InternVL3-2B** | Qwen2.5-1.5B base | Alibaba | 63.9 |
| **InternVL3-8B** | Qwen2.5-7B base | Alibaba | 73.3 |
| **InternVL3-9B** | InternLM3-8B | Shanghai AI Lab | 72.4 |
| **InternVL3-14B** | Qwen2.5-14B base | Alibaba | 75.5 |
| **InternVL3-38B** | Qwen2.5-32B base | Alibaba | 77.3 |
| **InternVL3-78B** | Qwen2.5-72B base | Alibaba | 79.5 |

[[entities/internvl-2-5|InternVL 2.5]] の InternLM 2.5 系から **Qwen 2.5 系へ大半シフト**（9B のみ InternLM3-8B）。**1B-8B は InternViT-300M-V2.5**、**38B-78B は InternViT-6B-V2.5**（InternVL 2.5 から同じ視覚エンコーダ）。

---

## 実験結果と知見

### MMMU 72.2 で SOTA を更新

| Model | MMMU(val) |
|---|---|
| InternVL2-Llama3-76B | 62.7 |
| InternVL2.5-78B | 70.0 |
| **InternVL3-78B** | **72.2** (+2.2) |
| GPT-4o-20241120 | 70.7 |
| Claude-3.7-Sonnet | 75.0 |
| Gemini-2.0-Pro | 69.9 |
| Qwen2.5-VL-72B | 68.2 |

**Claude-3.7-Sonnet 75.0 には届かないが、GPT-4o / Claude-3.5 / Qwen2.5-VL-72B / Gemini-2.0-Pro を超える**。**Test-Time Scaling（Best-of-8）で 72.2 のまま保持**（既に高水準で頭打ち）。

### 数学推論で商用と互角

InternVL3-78B: MathVista **79.0**（GPT-4o 60.0、Claude-3.7-Sonnet 66.8、Qwen2.5-VL-72B 74.2、Gemini-2.0-Pro 71.3）。**MathVerse 51.0**（Best-of-8 で 54.2）。

### 言語能力: 「マルチモーダル化で言語が強くなる」

**最も衝撃的な結果**（表 11）:

| Base LLM | Qwen2.5-Chat | InternVL3 | Δ |
|---|---|---|---|
| Qwen2.5-0.5B → 1B | 33.5 | **42.4** | **+8.9** |
| Qwen2.5-7B → 8B | 69.4 | **72.9** | **+3.5** |
| Qwen2.5-72B → 78B | 78.9 | **80.5** | **+1.6** |

**全 LLM サイズで Qwen2.5-Chat を上回る**（小型ほど顕著）。これは **「ネイティブマルチモーダル事前学習が言語能力も強化する」** という InternVL シリーズで初の本格的実証。**Qwen2.5-Chat は純粋言語タスクに特化して訓練された**にもかかわらず、**マルチモーダル要素を含む InternVL3 が上回る**。

### 空間推論: GPT-4o / Gemini を圧倒（VSI-Bench）

| Model | VSI-Bench Overall |
|---|---|
| GPT-4o | 34.0 |
| Gemini-1.5 Pro | 45.4 |
| **InternVL3-8B** | **42.1** |
| **InternVL3-38B** | **48.9** |
| **InternVL3-78B** | **48.4** |

**InternVL3-8B（8B）が GPT-4o を +8.1 圧倒**。InternVL3-38B が物体カウント 71.7 / 絶対距離推定 50.2 / 相対距離推定 53.5。**3D シーン理解の新水準**。

### GUI Grounding: UI-TARS と互角

ScreenSpot: **InternVL3-72B 88.7%**（UI-TARS-72B 88.4 / Qwen2.5-VL-72B 87.1 / GPT-4o 18.1）。
ScreenSpot-V2: **InternVL3-72B 90.9%**（UI-TARS-72B 90.3 超え）。

### OCRBench で 900 突破

**OCRBench 906**（InternVL3-78B）。Qwen2.5-VL-72B 885、InternVL2.5-78B 854。**史上初の 900 超え**。

### Visual Grounding で InternVL 2.5 にわずかに退行

| Model | RefCOCO Overall |
|---|---|
| **InternVL2.5-78B** | **92.3** |
| InternVL3-78B | 91.4 |

著者は **「訓練データ拡張で grounding 特化データが含まれず、相対的に grounding データ比率が低下したため」** と分析。シリーズ初の「先代に劣る」ケース、将来課題。

---

## 限界・批判的視点

### Claude-3.7-Sonnet との MMMU 差

- MMMU: InternVL3-78B 72.2 vs Claude-3.7-Sonnet 75.0 (-2.8)
- Overall reasoning: InternVL3-78B 54.6 vs Gemini-2.0-Pro 58.5 (-3.9)
- **最強商用フロンティアとの差は依然 3-4 ポイント**

### Visual Grounding の退行

- InternVL2.5-78B 92.3 → InternVL3-78B 91.4 (-0.9)
- データ拡張時に grounding データを増やさなかった選択の負の結果

### WildVision の改善幅は限定的

- InternVL2.5-78B 71.4 → InternVL3-78B 73.6 (+2.2)
- GPT-4o 80.6 にはまだ届かない（-7.0）
- 「長応答 + 人間嗜好整合」は依然弱点

### Native Pre-Training のスケーラビリティ未検証

- InternVL2-8B での置換実験のみ（表 3 図 3）。**ゼロから訓練（重み初期化なし）** の効果は未検証
- 「base LLM 初期化を維持」する限り、純粋 Native ではない

### Apache 2.0 のような完全自由ライセンスではない

- Qwen2.5 base に依存 → Tongyi Qianwen LICENSE 制約
- 商用利用に注意（特に大規模ユーザ）

### V2PE の効果は短文脈タスクでも軽微

- 表 12: V2PE なし 75.2 vs V2PE δ=1/4 75.9（+0.7）
- 「長文脈で大効果」を実証する評価が本論文には不足

---

## 用語と略称

| 略称 | 展開 | 意味 |
|---|---|---|
| **InternVL 3** | – | InternVL シリーズ第 7 世代（2025 Apr、本論文） |
| **InternVL3-1B/2B/8B/9B/14B/38B/78B** | – | InternVL 3 の 7 サイズ（9B は InternLM3-8B ベース） |
| **InternLM3-8B** | – | Shanghai AI Lab の最新 8B LLM（InternLM 2.5 の後継） |
| **Qwen2.5-0.5B/1.5B/7B/14B/32B/72B base** | – | Alibaba Qwen2.5 の **base モデル**（Instruct/Chat ではない） |
| **InternViT-300M-V2.5 / InternViT-6B-V2.5** | – | InternVL 2.5 と共有の視覚エンコーダ |
| **Native Multimodal Pre-Training** | – | テキスト + マルチモーダル共同事前学習（本論文の中核） |
| **V2PE** | Variable Visual Position Encoding | 視覚トークンに小さな位置インクリメント δ を使用 |
| **δ** | – | V2PE の視覚トークン位置インクリメント（{1, 1/2, 1/4, ..., 1/256}） |
| **MPO** | Mixed Preference Optimization | DPO + BCO + LM 損失の混合選好最適化 |
| **DPO** | Direct Preference Optimization | 人間選好直接最適化（Rafailov et al., 2023） |
| **BCO** | Binary Classifier Optimization | 個別応答の絶対品質を独立評価 |
| **VisualPRM** | Visual Process Reward Model | 推論ステップごとに +/- スコアを付与する critic モデル |
| **VisualPRM-8B** | – | InternVL3-8B ベースの 8B critic |
| **VisualPRM400K** | – | VisualPRM 訓練データ（MMPR v1.2 ベース） |
| **MMPR v1.2** | Multimodal Preference Repository v1.2 | 多モーダル選好データセット |
| **Best-of-N (Bo8)** | – | テスト時スケーリング: N=8 個の応答から最良選択 |
| **Test-Time Scaling** | – | 推論時計算を増やして性能向上（CoT + Best-of-N） |
| **SFT** | Supervised Fine-Tuning | 後訓練の第 1 段階（21.7M サンプル） |
| **CoT** | Chain-of-Thought | 段階的推論 |
| **Square Averaging** | – | $w_i = 1/l^{0.5}$ の損失重み付け（[[entities/internvl-2-5\|InternVL 2.5]] と同じ） |
| **Random JPEG Compression** | – | 品質 75-100 圧縮 augmentation |
| **Multimodal Data Packing** | – | 複数サンプルを長系列に連結 |
| **InternEVO** | – | InternVL 用に拡張された ZeRO 最適化訓練フレームワーク |
| **Head-Parallel** | – | 32K トークン系列対応の並列化技法 |
| **Sequence-Parallel** | – | 系列方向のテンソル並列 |
| **GUI Grounding** | – | スクリーンショット上の UI 要素の位置特定 |
| **ScreenSpot / ScreenSpot-V2** | – | GUI grounding ベンチマーク |
| **UI-TARS-72B** | – | ByteDance の GUI エージェント特化 MLLM |
| **Aguvis-72B** | – | GUI 特化 MLLM（ScreenSpot SOTA） |
| **VSI-Bench** | Visual-Spatial Intelligence Benchmark | 3D 空間推論ベンチマーク |
| **MathVision** | – | MATH-Vision の別名 |
| **MathVerse** | – | 視覚数学ベンチマーク（Vision-Only split を使用） |
| **DynaMath** | – | 動的数学推論ベンチマーク |
| **WeMath** | – | 視覚数学ベンチマーク |
| **LogicVista** | – | 論理推論ベンチマーク |
| **MMMU-Pro** | – | （表中で省略、本論文では MMMU のみ） |
| **CharXiv** | – | 科学論文 chart 理解（RQ/DQ） |
| **VCR-EN-Easy** | Visual Caption Restoration | 画像内テキスト復元 |
| **Mantis Eval** | – | 複数画像推論 |
| **MMIU / MuirBench / MIRB / MMT-Bench** | – | 複数画像理解ベンチ群 |
| **BLINK** | – | 視覚知覚ベンチ |
| **R-Bench** | – | 実世界画像劣化頑健性 |
| **CG-Bench** | – | clue-based 長動画理解 |
| **Tongyi Qianwen LICENSE** | – | Qwen2.5 のライセンス（商用制約あり） |
| **GLM-4v-Plus** | – | 智谱 AI の MLLM（比較相手） |
| **Step-1o** | – | StepFun の MLLM（比較相手） |
| **OpenCompass Academic** | – | OpenCompass マルチモーダル学術リーダーボード |
| **QvQ-72B-Preview** | – | Alibaba の reasoning-focused 視覚 MLLM |
| **Ovis2-16B / 34B** | – | Ovis シリーズ後継 |
| **Aquila-VL-2B** | – | 智源 BAAI の MLLM |
| **MiniCPM-o2.6** | – | MiniCPM-V 2.6 後継 |
| **Oryx-1.5-32B** | – | 動画 MLLM |
| **VideoLLaMA2-72B** | – | DAMO 動画 MLLM |
| **InternLM2-Math / Code** | – | InternLM2 数学・コード特化版 |
| **MMPR (Multimodal Preference Repository)** | – | 多モーダル選好データセット |
| **OmniCorpus** | – | 大規模多モーダル interleaved コーパス（Shanghai AI Lab） |
| **InternVL-Data** | – | InternVL3 公開訓練データセット |

---

## 関連ページ

- 翻訳: [[translations/internvl-3]]
- 主要エンティティ: [[entities/internvl-3]]
- 直接の祖: [[sources/internvl-2-5]] / [[entities/internvl-2-5]]（InternVL 2.5、商用追従ライン最終到達点）
- 視覚エンコーダ: [[entities/internvit-300m]]（InternViT-300M-V2.5 を共有）
- InternVL シリーズ系譜: [[entities/internvl]] / [[entities/internvl-1-5]] / [[entities/mini-internvl]] / [[entities/internvl-2-5]] / [[entities/internvl-3]]
- 関連視覚エンコーダ: [[entities/clip]] / [[entities/siglip]] / [[entities/perception-encoder]] / [[entities/dinov2]]
- 関連概念: [[concepts/foundation-model]], [[concepts/weakly-supervised-pretraining]], [[concepts/zero-shot-transfer]], [[concepts/alignment-tuning]]（PE の中間層特徴発見との対比）, [[concepts/knowledge-distillation]]
