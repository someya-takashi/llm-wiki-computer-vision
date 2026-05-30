---
type: entity
entity_kind: model
aliases: [MPO, Mixed Preference Optimization, InternVL2-8B-MPO]
tags: [preference-optimization, dpo, bco, post-training, mllm-cot, opengvlab]
related: [[concepts/foundation-model]], [[entities/mmpr]], [[entities/internvl-3]], [[entities/internvl-2-5]], [[entities/internvl]]
sources: [[sources/mpo]]
updated: 2026-05-29
---

# MPO — Mixed Preference Optimization アルゴリズム

## 概要

**MPO** = **Mixed Preference Optimization**。OpenGVLab Shanghai AI Lab が 2024 年 11 月に提案した、**MLLM の Chain-of-Thought 推論を強化する選好最適化アルゴリズム**。3 つの損失の線形結合:

$$\mathcal{L}_{\text{MPO}} = w_p \mathcal{L}_p + w_q \mathcal{L}_q + w_g \mathcal{L}_g$$

| 損失 | 採用 | 役割 |
|---|---|---|
| $\mathcal{L}_p$ | **DPO** | 応答ペア間の相対選好 |
| $\mathcal{L}_q$ | **BCO**（Binary Classifier Optimization） | 個別応答の絶対品質 |
| $\mathcal{L}_g$ | **SFT loss** | 選好応答の生成過程 |

- 論文: "Enhancing the Reasoning Ability of MLLMs via Mixed Preference Optimization"
- arXiv: 2411.10442（2024 Nov）
- 詳細: [[sources/mpo]] / 翻訳: [[translations/mpo]]
- プロジェクト: <https://internvl.github.io/blog/2024-11-14-InternVL-2.0-MPO/>

---

## 数学的定義

### 全体損失

$$\mathcal{L} = w_p \mathcal{L}_p + w_q \mathcal{L}_q + w_g \mathcal{L}_g$$

**最適ハイパーパラメータ**: $w_p = 0.8, w_q = 0.2, w_g = 1.0$。

### Preference Loss (DPO)

Bradley-Terry モデル仮定。報酬モデル不要:

$$\mathcal{L}_p = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_c | x)}{\pi_0(y_c | x)} - \beta \log \frac{\pi_\theta(y_r | x)}{\pi_0(y_r | x)}\right)$$

- $\pi_\theta$: 学習可能 policy
- $\pi_0$: 凍結 reference（初期重み）
- $\beta = 0.1$: KL ペナルティ係数

### Quality Loss (BCO)

chosen→1、rejected→0 へバイナリ分類:

$$\mathcal{L}_q = \mathcal{L}_q^+ + \mathcal{L}_q^-$$

$$\mathcal{L}_q^+ = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_c | x)}{\pi_0(y_c | x)} - \delta\right)$$

$$\mathcal{L}_q^- = -\log \sigma\left(-\left(\beta \log \frac{\pi_\theta(y_r | x)}{\pi_0(y_r | x)} - \delta\right)\right)$$

$\delta$: 過去報酬の moving average（安定化）。

### Generation Loss (SFT)

標準的な next-token prediction:

$$\mathcal{L}_g = -\frac{\log \pi_\theta(y_c | x)}{|y_c|}$$

---

## 訓練設定

| 項目 | 値 |
|---|---|
| **学習率** | 5e-6 (5e-5 で破綻、5e-7 で moderate) |
| **Batch size (global)** | 256 |
| **Optimizer** | AdamW (β₁=0.9, β₂=0.999, weight decay 0.05) |
| **LR schedule** | Linear warmup (5% steps) + Cosine decay (min 0) |
| **KL penalty β** | 0.1 |
| **Weights** | $w_p = 0.8, w_q = 0.2, w_g = 1.0$ |
| **Epochs** | 1 |
| **Initialization** | InternVL2-8B（または該当 MLLM）から |
| **Trainable params** | 全パラメータ |

---

## なぜ DPO + BCO + SFT loss なのか（アブレーション結果）

論文では 10 個の PO アルゴリズム（DPO / RSO / IPO / cDPO / RobustDPO / BCO / SPPO / AOT / TR-DPO / ORPO）を比較し、以下を発見:

### Vanilla アルゴリズム（M3CoT、SFT 損失なし）

| Method | Direct | CoT | Δ |
|---|---|---|---|
| InternVL2-8B | 59.3 | 57.0 | **-2.3** |
| SFT | 65.7 | 68.5 | +2.8 |
| DPO | 75.8 | 72.7 | **-3.1** |
| RSO | 74.2 | 74.3 | +0.1 |
| BCO | **78.1** | **78.4** | +0.3 |
| TR-DPO | 75.9 | 66.8 | **-9.1** |

**DPO 単独は CoT 改善せず**（むしろ悪化）。BCO は最良だが、CoT 改善幅が小さい。

### SFT 損失拡張版（X+）

| Method | Direct | CoT | Δ |
|---|---|---|---|
| ORPO | 66.6 | 73.9 | +7.3 |
| DPO+ | 76.4 | 78.9 | +2.5 |
| BCO+ | 77.4 | 78.4 | +1.0 |
| AOT+ | 76.3 | 78.0 | +1.7 |
| **MPO (DPO+BCO+SFT)** | **77.7** | **79.1** | +1.4 |

**MPO が CoT で最高スコア 79.1**。SFT 損失追加が CoT 強化の鍵。

### 主要観察

1. **DPO 単体は CoT で改善しない** → 報酬ハッキング的に gibberish 出力
2. **SFT loss が CoT 強化の鍵** → 全 PO に追加すると CoT 改善
3. **参照モデル凍結が重要** → TR-DPO（参照更新あり）は CoT で大幅悪化
4. **参照モデル制約が重要** → ORPO（参照不要）は SFT 拡張版より悪い
5. **DPO + BCO が最良の組み合わせ** → MPO へ

---

## 主要結果: InternVL2-8B-MPO

### 推論ベンチマーク（vs InternVL2-8B）

| ベンチ | Baseline | InternVL2-8B-MPO | Δ |
|---|---|---|---|
| **M3CoT** | 59.3 | **79.2** | **+19.9** |
| **MathVista** | 58.3 | **67.0** | **+8.7** |
| **MathVision** | 20.4 | **25.7** | **+5.3** |
| MMVet | 54.2 | 56.2 | +2.0 |
| LLaVA-Bench | 73.2 | 76.7 | +3.5 |

### ハルシネーション削減

| ベンチ | Baseline | InternVL2-8B-MPO | Δ |
|---|---|---|---|
| POPE | 86.9 | 88.1 | +1.2 |
| CRPE | 75.0 | 75.4 | +0.4 |
| MMHal-Bench | 3.3 | 3.5 | +0.2 |

### CoT 性能の変化

| Mode | Baseline | InternVL2-8B-MPO |
|---|---|---|
| **M3CoT Direct** | 59.3 | **77.2** |
| **M3CoT CoT** | 57.0 | **79.2** |
| **Δ (CoT - Direct)** | **-2.3** | **+2.0** |

**Baseline は CoT で悪化、MPO は CoT で改善する**。

### テキスト専用ベンチマーク（13 ベンチ平均）

**MMPR にテキスト専用データなし**にもかかわらず:

| Setting | Average | TheoremQA | IFEval |
|---|---|---|---|
| Baseline | 62.1 | 15.6 | 52.3 |
| SFT | 62.0 | 15.8 | 53.6 |
| **MPO** | **63.0** | **20.8** | **56.4** |

**マルチモーダル選好最適化が言語推論能力も強化** → [[entities/internvl-3|InternVL 3]] の「Native Multimodal Pre-Training で言語が強くなる」発見の先例。

---

## InternVL シリーズでの位置づけ

```
[InternVL 2 (2024-07)]
   訓練: pre-training + SFT
   問題: CoT 推論で性能低下

[InternVL2-8B-MPO (2024-11、本ページ)]
   訓練: + MPO (Stage 3)
   結果: CoT で改善するモデルに変身

[InternVL 2.5 (2024-12)]
   MPO は採用されず、data filtering + Test-Time Scaling が主軸

[InternVL 3 (2025-04)] ([[entities/internvl-3]])
   訓練: Native Multimodal Pre-Training + SFT + MPO (Stage 3)
   ★ MPO が正式採用、+4.1 ~ +4.5 ポイント推論改善（38B/78B）
   MMPR v1.2（拡張版）を使用、300K 選好ペア

[InternVL 3.5 (2025-08)] ([[entities/internvl-3-5]])
   訓練: SFT + Cascade RL (Stage 1: MPO + Stage 2: GSPO)
   ★ MPO が「Cascade RL の offline warm-up 段階」として進化
   MPO 単独 → +3-5 ポイント、Cascade RL (MPO + GSPO) → +6-12 ポイント
   GSPO 単独の半分の GPU 時間で +2.1 ポイント上回る
```

**InternVL 3 で MPO は正式採用され、InternVL 3.5 で Cascade RL の Stage 1 として進化**。**InternVL シリーズの永続的技術**。

### InternVL 3 での MPO 効果（vs SFT のみ）

| Model | w/o MPO | w/ MPO | Δ |
|---|---|---|---|
| InternVL3-1B | 24.6 | 25.1 | +0.5 |
| InternVL3-8B | 41.4 | 44.3 | **+2.9** |
| InternVL3-14B | 46.2 | 49.9 | **+3.7** |
| **InternVL3-38B** | **48.3** | **52.8** | **+4.5** |
| **InternVL3-78B** | **50.5** | **54.6** | **+4.1** |

**大型モデルほど MPO の効果大**。本論文の 8B での観察が大型モデルにスケールすることを確認。

---

## DPO 系統との比較

| Algorithm | Reference Model | Reward Model | 安定性 | CoT 性能 |
|---|---|---|---|---|
| RLHF (PPO) | 必要 | 必要 | 不安定 | – |
| DPO | 必要（凍結） | 不要 | 中程度 | 弱い |
| IPO | 必要（凍結） | 不要 | 中程度 | 弱い |
| BCO | 必要（凍結） | 不要 | 良 | 中程度 |
| TR-DPO | 必要（更新あり） | 不要 | 不安定 | **大幅悪化** |
| ORPO | **不要** | 不要 | 良 | 弱い |
| **MPO** | **必要（凍結）** | **不要** | **良** | **最良** |

**MPO の独自貢献**: 「DPO の相対選好 + BCO の絶対品質 + SFT の生成過程」の **3 要素同時最適化** が、MLLM の CoT 推論において他の組み合わせより優れる。

---

## 公開リソース

公開済み（HuggingFace `OpenGVLab/`）:

| モデル | 詳細 |
|---|---|
| `InternVL2-8B-MPO` | InternVL2-8B + MMPR + MPO |
| `MMPR-v1.2` | 拡張版（InternVL 3 で使用） |

**ライセンス**: MIT + InternLM2

---

## 関連ページ

- 詳細解説: [[sources/mpo]]
- 翻訳: [[translations/mpo]]
- 関連データセット: [[entities/mmpr]]（MPO 訓練用選好データセット）
- 主な応用先: [[entities/internvl-3]]（Stage 3 で正式採用、+4.1 ~ +4.5 ポイント推論改善）
- InternVL シリーズ: [[entities/internvl]] / [[entities/internvl-2-5]] / [[entities/internvl-3]]
- 関連概念: [[concepts/foundation-model]]
