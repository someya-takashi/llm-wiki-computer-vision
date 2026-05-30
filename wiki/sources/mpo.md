---
type: source
source_path: raw/papers/Enhancing the Reasoning Ability of Multimodal Large Language Models via Mixed Preference Optimization.md
source_kind: paper
title: "Enhancing the Reasoning Ability of Multimodal Large Language Models via Mixed Preference Optimization"
authors: [Weiyun Wang, Zhe Chen, Wenhai Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Jinguo Zhu, Xizhou Zhu, Lewei Lu, Yu Qiao, Jifeng Dai]
year: 2024
venue: "arXiv:2411.10442 (technical report)"
ingested: 2026-05-29
tags: [mllm, preference-optimization, dpo, bco, mpo, mmpr, dropoutntp, cot, internvl-2-8b-mpo, opengvlab]
translation: [[translations/mpo]]
---

# MPO — Mixed Preference Optimization と MMPR データセット、CoT 推論を救った論文

> 原典: [[translations/mpo]] ・ `raw/papers/Enhancing the Reasoning Ability of Multimodal Large Language Models via Mixed Preference Optimization.md`
> 著者: Weiyun Wang, Zhe Chen, Wenhai Wang ら（OpenGVLab Shanghai AI Lab + 復旦大学 + 南京大学 + 香港中文大学 + 清華大学 + SenseTime Research）
> 投稿: arXiv:2411.10442（2024 年 11 月、technical report）
> プロジェクトページ: <https://internvl.github.io/blog/2024-11-14-InternVL-2.0-MPO/>

---

## 一言まとめ

**「オープン MLLM の最大の弱点 = CoT 推論で性能が低下する」 という現象を、**Preference Optimization で初めて本格的に解決した論文**」**。著者らは **「SFT が teacher forcing による分布シフトを生み、長い CoT rationale を生成すると蓄積誤差で性能が落ちる」** と分析、**Mixed Preference Optimization（MPO = DPO + BCO + SFT loss の混合）+ DropoutNTP（画像なしで応答を補完して rejected 生成）+ MMPR（3M 選好データ）** という 3 点セットを提案。**InternVL2-8B-MPO が MathVista で +8.7（67.0、10× 大きな InternVL2-76B と同等）、M3CoT で +21.9（59.3 → 79.2）** という劇的改善。この MPO は [[entities/internvl-3|InternVL 3]] の Stage 3（post-training）として正式採用され、**InternVL シリーズの CoT 推論能力の中核技術** となった。

---

## 背景と問題意識

### 「CoT で MLLM が悪化する」謎の現象

論文の図 1 が示す衝撃的な事実: **多くのオープン MLLM（LLaVA-1.5-13B, Qwen2-VL-7B, MiniCPM-V-2.6-8B, InternVL2-8B 等）が、Chain-of-Thought（CoT）で推論すると、直接答える場合より性能が低下する**。

```
[InternVL2-8B の MathVista]
   直接応答: 58.3
   CoT 応答: 56.8 (-1.5)
   ↑
   「ステップごとに考えると、なぜか悪くなる」
```

LLM の世界では「CoT は推論を改善する」が常識だった。**MLLM ではなぜ逆になるのか？**

### 著者らの分析: SFT による分布シフト

著者らはこの現象を **SFT 損失が導入する分布シフト** と説明:

```
[SFT の構造的問題]
   訓練時: teacher forcing（前の正解トークンに基づき次を予測）
   推論時: モデル自身の出力に基づき次を予測

   → 短い直接応答ならまだ大丈夫
   → 長い CoT 応答では誤差が蓄積、分布が崩壊
   → CoT で性能低下！
```

これは **「LLM の CoT 成功と MLLM の CoT 失敗」のギャップ** を説明する重要な観察。

### この論文の位置づけ

**「MLLM 訓練パイプラインの第 3 段階（post-training）の確立」**。LLaVA 系の MLLM は事前学習 + SFT の 2 段階で完結していたが、**LLM 界では DPO/RLHF が既に標準** だった。この論文は **その方法論を MLLM へ持ち込み、CoT 推論の問題を解決した先駆け**。

### 既存マルチモーダル選好データの限界

| データセット | 主目的 | 問題 |
|---|---|---|
| RLAIF-V | ハルシネーション削減 | 自然画像 + 知覚データのみ、**推論データなし** |
| Silkie / POVID / RLHF-V | ハルシネーション削減 | 同様 |

**「マルチモーダル推論用の選好データがない」** という大きな空白を、本論文が埋める。

---

## 提案手法

### 1. MMPR データセット（約 3M 選好サンプル）

**MultiModal PReference dataset**。**正解あり 2.5M + 正解なし 750K = 約 3M サンプル**。

**6 ドメイン × 18 データセット** から収集:

| Task | Datasets |
|---|---|
| **General VQA** | VQAv2, GQA, OKVQA, IconQA |
| **Science** | AI2D, ScienceQA, M3CoT |
| **Chart** | ChartQA, DVQA, MapQA |
| **Mathematics** | GeoQA+, CLEVR-Math, Geometry3K, GEOS, GeomVerse, Geo170K |
| **OCR** | OCRVQA, InfoVQA, TextVQA, STVQA, SROIE |
| **Document** | DocVQA |

#### 2 つのパイプライン

**(A) 正解あり指示 → 正誤判定ベース**:
```
1. InternVL2-8B から 32 個の解答をサンプリング（CoT 形式、「Final Answer: ***」で締める）
2. 正解と一致 → chosen (positive)
3. 不一致 / 最終答えなし → rejected (negative)
4. 1 クエリあたり最大 15 ペア構築
```

**(B) 正解なし指示 → DropoutNTP（本論文の独自貢献）**:
```
1. InternVL2-8B から解答 y を生成
2. y を半分で切り詰める → y_<j（前半）, y_>=j（後半）
3. InternVL2-8B に y_<j のみを渡し、画像なしで残りを補完 → ~y_>=j
4. chosen = [y_<j, y_>=j] = y（オリジナル）
   rejected = [y_<j, ~y_>=j]（画像なしで補完されたもの、ハルシネーション含む）
```

> **DropoutNTP の天才的なアイデア**: 「画像入力なしで応答を続けると、モデルは画像の中身を **推測（hallucination）** で補完する。その推測は元応答より**必ず悪い**」という構造的な性質を利用。**人手アノテーション不要で大規模に rejected を生成** できる。

**RLAIF-V との比較**: 
- RLAIF-V: divide-and-conquer 方式（応答を atomic claims に分解、各々を検証）、選好ペアあたり **992.7 トークン**
- DropoutNTP: **571.2 トークン**（**RLAIF-V の 57.5%**）、性能はほぼ同等（Object HalBench 7.3 vs 7.6、MMHal-Bench 3.5 vs 3.6）

### 2. MPO アルゴリズム（本論文の中核）

**重要な観察**: 「DPO 単独で大規模選好データを学習すると、**合理的な rationale を生成できず gibberish を出力する**」（Smaug の分析と一致）。

著者らは **「効果的な PO プロセスは 3 つを学習すべき」** と主張:
1. **応答ペア間の相対選好**（preference loss）
2. **個別応答の絶対品質**（quality loss）
3. **選好応答の生成プロセス**（generation loss）

**MPO 損失**:
$$\mathcal{L} = w_p \mathcal{L}_p + w_q \mathcal{L}_q + w_g \mathcal{L}_g$$

| 損失 | 役割 | 採用アルゴリズム |
|---|---|---|
| $\mathcal{L}_p$ | **相対選好**: chosen を rejected より好む | **DPO** |
| $\mathcal{L}_q$ | **絶対品質**: chosen→1、rejected→0 のバイナリ分類 | **BCO**（Binary Classifier Optimization） |
| $\mathcal{L}_g$ | **生成過程**: 標準的な next-token prediction | **SFT loss** |

**ハイパーパラメータ**（実装詳細から）:
- $w_p = 0.8$、$w_q = 0.2$、$w_g = 1.0$
- 学習率 5e-6（5e-5 で大幅悪化、5e-7 で moderate）
- KL ペナルティ β = 0.1
- 1 epoch、global batch size 256

#### なぜ DPO + BCO + SFT loss なのか（アブレーションの結論）

10 個の PO アルゴリズム（DPO / RSO / IPO / cDPO / RobustDPO / BCO / SPPO / AOT / TR-DPO / ORPO）を比較した結果:

| 観察 | 含意 |
|---|---|
| ほぼ全 PO が SFT を Direct で上回る | PO はマルチモーダル推論の改善に有効 |
| **DPO 単体は CoT で改善しない** | DPO だけでは長い CoT に対応できない |
| **SFT loss を加えると全アルゴリズムが CoT 強化** | **SFT loss が CoT 推論強化の鍵** |
| TR-DPO（参照モデル更新）は CoT で大幅悪化 | **参照モデル凍結が重要** |
| ORPO（参照モデル不要）は SFT 拡張版より悪い | **参照モデル制約が重要** |
| **DPO+ と BCO+ が最良 CoT 性能** | この 2 つを統合 → MPO |

### 3. マルチモーダル CoT の 3 種の精緻化

データサンプリング時に **3 種類の CoT** を併用:

1. **Background Knowledge-based CoT**（Science）: 関連背景知識 → 推論 → 最終答え
2. **Visual Content-based CoT**（Chart / OCR / Document）: 視覚的内容分析 → 推論 → 最終答え
3. **Grounded CoT**（General VQA）: 応答内のオブジェクトを画像領域にリンク

これらは「データ多様性」+ **「DropoutNTP の rejected 品質改善」** に効く（背景知識を冒頭に置くと、画像なしの補完も似た構造を持ちやすくなり、極端な品質差を防ぐ）。

---

## 実験結果と知見

### MathVista で +8.7 ポイント、76B モデルと同等

**InternVL2-8B-MPO の結果**（vs InternVL2-8B）:

| ベンチ | InternVL2-8B | **InternVL2-8B-MPO** | Δ |
|---|---|---|---|
| **M3CoT** | 59.3 | **79.2** | **+19.9** |
| **MathVista** | 58.3 | **67.0** | **+8.7** |
| **MathVision** | 20.4 | **25.7** | **+5.3** |
| MMVet | 54.2 | 56.2 | +2.0 |
| LLaVA-Bench | 73.2 | 76.7 | +3.5 |
| POPE | 86.9 | 88.1 | +1.2 |
| CRPE | 75.0 | 75.4 | +0.4 |
| MMHal-Bench | 3.3 | 3.5 | +0.2 |

**注目**: 
- **MathVista 67.0 = InternVL2-76B の 67.2 とほぼ同等**（10× サイズ差を吸収）
- **MathVision 25.7 で当時オープン MLLM 新 SOTA**

### CoT で性能が改善する MLLM に変身

| Model | Setting | M3CoT | MathVista | MMVet | POPE |
|---|---|---|---|---|---|
| InternVL2-8B | Direct | 59.3 | 58.3 | 54.2 | 86.9 |
| InternVL2-8B | CoT | **57.0 ↓** | **56.8 ↓** | 54.7 | **82.9 ↓** |
| SFT | Direct | 63.9 | 62.7 | 54.7 | 86.5 |
| SFT | CoT | 67.8 | 64.2 | **53.8 ↓** | **84.0 ↓** |
| **MPO** | **Direct** | **77.2** | **64.5** | **55.1** | **87.0** |
| **MPO** | **CoT** | **79.2** | **67.0** | **56.2** | **88.1** |

**InternVL2-8B-MPO は全タスクで CoT が direct より良い** = **CoT が推論を改善するモデルになった**。

### テキスト専用ベンチマークでも改善（マルチモーダル訓練だけで！）

**「MMPR にはテキスト専用データが含まれない」**にもかかわらず、MPO 訓練モデルが純粋言語タスクで baseline を **+0.9 ポイント上回る**（平均 13 ベンチ）。

| Benchmark | Baseline | SFT | **MPO** |
|---|---|---|---|
| TheoremQA | 15.6 | 15.8 | **20.8** (+5.2) |
| IFEval | 52.3 | 53.6 | **56.4** (+4.1) |
| Average | 62.1 | 62.0 | **63.0** (+0.9) |

**「マルチモーダル選好最適化が言語推論能力も強化する」** という、後の [[sources/internvl-3|InternVL 3]] の「Native Multimodal Pre-Training で言語が強くなる」発見につながる重要な先例。

### DropoutNTP のアブレーション

| Dropout Ratio | Object HalBench Resp.(↓) | MMHal-Bench Hall.(↓) |
|---|---|---|
| 0.25 | 9.3 | 40.6 |
| **0.50** | **7.6** | **31.3** |
| 0.75 | 11.6 | 36.5 |

**半分で切り詰めるのが最適**。0.25 だと rejected の大半が画像なし補完 = 品質差が大きすぎ訓練効果低下。0.75 だと共通プレフィックスが多すぎ = 品質差が見えにくい。

### データスケーリング

10K → 100K で **Direct/CoT 両方で性能向上**、特に **CoT 性能が顕著に向上**。**「選好データは多ければ多いほど CoT が良くなる」**。

### MPO のハイパーパラメータ

- 学習率: 5e-7 で moderate、**5e-6 で最良**、5e-5 で破綻
- **$w_p = 0.8, w_q = 0.2$** が最良
- **$w_g$ は重要**: $w_g = 0.01$（SFT loss を弱める）と CoT が direct より悪化、**SFT loss が CoT 強化に必須**

---

## 限界・批判的視点

### 「事後的」アプローチ

- MPO は **SFT 後の追加段階** として実施される。**[[sources/internvl-3|InternVL 3]] の Native Multimodal Pre-Training のような根本的解決ではない**
- InternVL 3 では「事前学習段階から共同最適化」が試みられ、MPO は **Stage 3（post-training）** として併用される形

### 報酬モデル不要だが、参照モデルが必要

- DPO 系統の課題: **参照モデル $\pi_0$（凍結）** が訓練中ずっと必要 = **メモリ 2 倍**
- TR-DPO（参照モデル更新あり）や ORPO（参照モデル不要）を試したが、いずれも MPO に劣ることが分かった

### MMPR の構成の偏り

- VQA + Science + Math + Chart + OCR + Document の 6 ドメインのみ
- **動画 / 3D / GUI 等は含まれない**（後の [[sources/internvl-3|InternVL 3]] で補完）

### DropoutNTP の限界

- 「画像なしで補完すると悪くなる」という前提は、**画像が必要な質問にのみ有効**
- 純粋なテキスト推論問題（例: Wikipedia 知識）には適用困難
- 「画像が決定的に重要な質問」というデータの選別が必要

### スケーラビリティ

- InternVL2-8B での実験のみ。**より大きいモデル（76B 等）への適用** はこの論文では未検証
- InternVL 3 で 1B-78B 全サイズに適用されたが、本論文時点では未検証

### MathVista 67.0 の限界

- 当時の InternVL2-76B 67.2 と同等で印象的だが、**商用 GPT-4o 63.8 や Gemini-1.5-Pro 63.9** とも近い
- 後の **Claude-3.5-Sonnet 67.7 / [[sources/internvl-2-5|InternVL 2.5-78B]] 72.3 / [[sources/internvl-3|InternVL 3-78B]] 79.0** には大きく劣る → MPO 単独では商用フロンティアには届かない

---

## InternVL シリーズでの位置づけ

**MPO は InternVL シリーズの「post-training」段階の中核技術**:

```
[InternVL 2 (2024-07)]
   訓練: pre-training + SFT
   問題: CoT が苦手、長い rationale で性能低下

[本論文 InternVL2-8B-MPO (2024-11)]
   訓練: pre-training + SFT + MPO（追加段階）
   結果: MathVista 67.0、CoT で性能改善するモデルに変身

[InternVL 2.5 (2024-12)]
   訓練: pre-training + SFT + Test-Time Scaling (CoT + Majority Voting)
   MPO は本格採用されず（データフィルタリングと test-time scaling が主軸）

[InternVL 3 (2025-04)] ([[entities/internvl-3]])
   訓練: Native Multimodal Pre-Training + SFT + MPO (Stage 3)
   MPO が正式採用、+4.1～+4.5 ポイント推論改善（38B/78B）
   MMPR v1.2（本論文の MMPR の拡張版）を使用
```

**「InternVL 2 系列の弱点（CoT）を本論文が解決し、InternVL 3 で正式採用された」** という流れ。MPO は **InternVL シリーズの DNA に組み込まれた永続的技術** となった。

---

## 用語と略称

| 略称 | 展開 | 意味 |
|---|---|---|
| **MPO** | Mixed Preference Optimization | DPO + BCO + SFT loss の混合（本論文の中核） |
| **MMPR** | MultiModal PReference dataset | 3M 規模のマルチモーダル選好データセット |
| **DropoutNTP** | Dropout Next-Token Prediction | 画像なしで応答を補完して rejected を生成（本論文の独自貢献） |
| **PO** | Preference Optimization | 選好最適化（DPO / IPO / BCO 等の総称） |
| **RLHF** | Reinforcement Learning from Human Feedback | 人間フィードバックによる強化学習 |
| **DPO** | Direct Preference Optimization | Rafailov et al. 2023、Bradley-Terry モデル仮定で報酬モデル不要 |
| **BCO** | Binary Classifier Optimization | Jung et al. 2024、chosen→1 / rejected→0 のバイナリ分類 |
| **PPO** | Proximal Policy Optimization | Schulman et al. 2017、RLHF の標準アルゴリズム |
| **SFT** | Supervised Fine-Tuning | 教師あり微調整 |
| **CoT** | Chain-of-Thought | 段階的推論（"Let's think step by step"） |
| **Teacher Forcing** | – | SFT で正解トークンを与えて訓練する手法 |
| **Distribution Shift** | – | 訓練時と推論時の分布のズレ |
| **Bradley-Terry model** | – | 1952 年提案、ペア比較から選好順序を推定する確率モデル |
| **Reference Model $\pi_0$** | – | DPO 系で凍結される元モデル（KL 制約用） |
| **Policy Model $\pi_\theta$** | – | DPO 系で更新される学習可能モデル |
| **KL Penalty $\beta$** | – | DPO の reference からの逸脱罰則係数（本論文では 0.1） |
| **Reward Shift $\delta$** | – | BCO の安定化用、過去報酬の moving average |
| **InternVL2-8B-MPO** | – | InternVL2-8B に MMPR + MPO を適用したモデル |
| **Object HalBench** | – | object-level hallucination ベンチマーク |
| **MMHal-Bench** | – | 96 問の hallucination 評価（GPT-4o 採点、0-6 スコア） |
| **POPE** | Polling-based Object Probing Evaluation | 物体ハルシネーション評価 |
| **CRPE** | Compositional Reasoning and Perception Evaluation | オブジェクト関係 hallucination 評価 |
| **M3CoT** | Multi-domain Multi-step Multi-modal CoT | マルチモーダル CoT ベンチマーク |
| **MathVista** | – | マルチモーダル数学推論ベンチマーク |
| **MathVision** | – | 実競技数学問題（より難しい） |
| **MMVet / MMVet v2** | – | 視覚汎用能力ベンチマーク |
| **LLaVA-Bench** | – | LLaVA 系の対話・記述・推論ベンチ |
| **RLAIF-V** | – | Yu et al. 2024、divide-and-conquer でマルチモーダル選好データ構築 |
| **Silkie / POVID / RLHF-V** | – | マルチモーダル選好データ手法（ハルシネーション特化） |
| **MMPR v1.2** | – | MMPR の InternVL 3 用拡張版（[[entities/internvl-3]] が使用） |
| **VisualPRM-8B** | – | InternVL3 の Process Reward Model（MMPR ベース、[[entities/internvl-3]] 参照） |
| **Smaug** | – | Pal et al. 2024、DPO が gibberish を生む現象を分析 |
| **RSO** | Statistical Rejection Sampling Optimization | DPO の hinge 損失版 |
| **IPO** | Identity Preference Optimization | DPO の overfitting 対策版 |
| **cDPO / RobustDPO** | – | preference label のノイズに頑健な DPO 変種 |
| **SPPO** | Self-Play Preference Optimization | 反復的に chosen→1/2, rejected→-1/2 へ push |
| **AOT** | Distributional Preference Alignment via Optimal Transport | optimal transport ベース |
| **TR-DPO** | – | reference model を定期的に同期する DPO 変種 |
| **ORPO** | Odds Ratio Preference Optimization | reference model 不要、NLL に log odds ratio 罰則 |
| **MMPR トークン数** | – | 正解なし: chosen 211.4 / rejected 171.2 平均、正解あり: chosen 300.0 / rejected 350.5 平均 |
| **token cost / pair** | – | DropoutNTP 571.2 vs RLAIF-V 992.7（57.5% コスト） |
| **VQAv2 / GQA / OKVQA / IconQA** | – | General VQA データソース |
| **AI2D / ScienceQA / M3CoT** | – | Science データソース |
| **ChartQA / DVQA / MapQA** | – | Chart データソース |
| **GeoQA+ / CLEVR-Math / Geometry3K / GEOS / GeomVerse / Geo170K** | – | Mathematics データソース |
| **OCRVQA / InfoVQA / TextVQA / STVQA / SROIE** | – | OCR データソース |
| **DocVQA** | – | Document データソース |
| **TheoremQA** | – | 複雑科学問題（テキスト専用ベンチ、MPO で +5.2） |
| **IFEval** | Instruction-Following Evaluation | 指示追従評価（MPO で +4.1） |
| **InternVL2-Pro** | – | InternVL2 の非公開最強モデル |

---

## 関連ページ

- 翻訳: [[translations/mpo]]
- 主要エンティティ: [[entities/mpo]]、[[entities/mmpr]]
- 直接の応用先: [[entities/internvl-3]] / [[sources/internvl-3]]（InternVL 3 Stage 3 で正式採用、MMPR v1.2 + 300K 選好ペア使用）
- InternVL シリーズ: [[entities/internvl]] / [[entities/internvl-1-5]] / [[entities/mini-internvl]] / [[entities/internvl-2-5]] / [[entities/internvl-3]]
- 関連概念: [[concepts/foundation-model]], [[concepts/zero-shot-transfer]], [[concepts/alignment-tuning]]
