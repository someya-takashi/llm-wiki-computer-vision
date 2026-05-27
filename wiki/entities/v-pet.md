---
type: entity
entity_kind: model
aliases: [V-PET, VFM-PEFT Ensemble Training]
tags: [semi-supervised-learning, self-training, pseudo-label-ensemble, parameter-efficient-fine-tuning, peft, lora, adaptformer, clip, dinov2, vfm, mean-labels, ohio-state]
related: [[concepts/semi-supervised-learning]], [[concepts/parameter-efficient-fine-tuning]], [[concepts/foundation-model]], [[entities/clip]], [[entities/dinov2]], [[entities/fixmatch]], [[entities/flexmatch]], [[sources/revisiting-ssl-foundation-models]]
sources: [[sources/revisiting-ssl-foundation-models]]
updated: 2026-05-28
---

# V-PET（VFM-PEFT Ensemble Training）

## 概要

**V-PET** は Ohio State University の Ping Zhang・Zheda Mai らが NeurIPS 2025 で発表した、Vision Foundation Models 時代向けの半教師あり学習アルゴリズム。「**複数の (VFM, PEFT) ペアで個別に fine-tune し、それぞれが生成する one-hot 疑似ラベルを平均（Mean Labels）してアンサンブル**」というシンプルな設計で、FixMatch/FlexMatch/SoftMatch/FineSSL を凌駕した。

- 論文: "Revisiting Semi-Supervised Learning in the Era of Foundation Models"
- arXiv: 該当（NeurIPS 2025 採択）
- 会議: NeurIPS 2025
- 所属: The Ohio State University
- コード: https://github.com/OSU-MLB/SSL-Foundation-Models

---

## アルゴリズム（4 フェーズ）

```
入力: ラベルあきデータセット L、ラベルなしデータセット U
      N 個の PEFT 手法、M 個の VFM
      初期パラメータ θ_{n,m}

出力: 最適パラメータ θ*

────── フェーズ (a) Supervised PEFT ──────
for n ∈ [1, N], m ∈ [1, M]:
    θ̃_{n,m} = θ_{n,m} を L で fine-tune

────── フェーズ (b) Pseudo-Label Generation ──────
for n ∈ [1, N], m ∈ [1, M]:
    P_{n,m} = { one_hot(argmax_c f_{θ̃_{n,m}}(u)) | u ∈ U }

────── フェーズ (c) Pseudo-Label Ensemble (Mean Labels) ──────
ソフト疑似ラベル: p̄_i = (1/(N·M)) Σ_{n,m} P_{n,m}[i]

────── フェーズ (d) Self-Training ──────
(n*, m*) を選択 → P 上で θ_{n*,m*} を fine-tune → θ*
```

**バリアント**:
- **V-PET**: 異なる VFM × 異なる PEFT のフルアンサンブル（最強）
- **PET**: 単一 VFM × 異なる PEFT のアンサンブル
- **ST**: 単一 (VFM, PEFT) ペアのみ（ベースライン）

---

## 主要なハイパーパラメータ

| パラメータ | デフォルト | 意味 |
|---|---|---|
| τ（信頼度閾値） | **0**（全疑似ラベル使用） | FixMatch とは正反対。アンサンブルで品質確保 |
| 自己訓練ラウンド数 | **1** | 反復しない |
| アンサンブルサイズ | 推奨 **2〜4** | それ以上は収穫逓減 |
| アンサンブル戦略 | **Mean Labels** | one-hot 化してから平均 |
| オプティマイザ | AdamW | バッチサイズ 32、weight decay 5e-4 |
| エポック数 | 35 | 単純設定 |

---

## 「Mean Labels が最良」の理由

異なる fine-tuned VFM の予測確率分布は**エントロピーが大きく異なる**（図 4）。たとえば DINOv2+AdaptFormer は高信頼予測が多く（低エントロピー）、CLIP+AdaptFormer は不確実な予測が多い（高エントロピー）。

| 戦略 | 問題点 |
|---|---|
| Mean Logits | スケールが揃わず、強モデルが支配 |
| Mean Probabilities | 同上、確率値の絶対値で支配される |
| **Mean Labels（one-hot 化後の平均）** | 全モデルを「**投票**」として等価扱い → 多数決による堅牢化 |

---

## 主要な実験結果（表 3、12 設定平均、6 VTAB データセット）

| 手法 | CLIP/LoRA | CLIP/Adapt | DINOv2/LoRA | DINOv2/Adapt |
|---|---|---|---|---|
| Labeled Only（fine-tune のみ） | 55.7 | 55.6 | 56.0 | 51.6 |
| FixMatch | 53.7 | 53.7 | 47.6 | 47.7 |
| FlexMatch | 56.2 | 53.9 | 48.6 | 47.8 |
| SoftMatch | 59.7 | 56.3 | 50.2 | 48.5 |
| FineSSL | 53.9 | 51.6 | 49.8 | 46.8 |
| ST（アンサンブルなし self-training） | 54.7 | 56.7 | 52.9 | 52.5 |
| PET（PEFT のみアンサンブル） | 59.3 | 59.7 | 56.7 | 58.0 |
| **V-PET（VFM × PEFT 両方アンサンブル）** | **60.5** | **61.0** | **61.2** | **61.5** |

**衝撃的な観察**: DINOv2 上では FixMatch/FlexMatch/SoftMatch/FineSSL が**Labeled Only より悪化**。SSL がむしろ性能を下げる。

---

## ベンチマーク：6 つの VTAB データセット

凍結 VFM が苦戦する領域から選定（CIFAR/Food101 のような「VFM が既に高精度」のベンチマークを意図的に避けた）：

| データセット | カテゴリ | ドメイン | クラス | ラベル/クラス |
|---|---|---|---|---|
| DTD | Natural | テクスチャ | 47 | 3, 6 |
| SUN397 | Natural | シーン | 397 | 3, 6 |
| RESISC45 | Specialized | リモートセンシング | 45 | 1, 2 |
| Retinopathy | Specialized | 医療画像 | 5 | 4, 8 |
| CLEVR-C | Structured | 合成推論 | 8 | 1, 2 |
| KITTI | Structured | 自動運転 | 4 | 5, 10 |

---

## 公正なハイパラチューニング：7 教師なし基準

ラベル希少な SeSL でデータリークを避けるため、**ラベルを使わずに**ハイパラを選ぶ手法を確立：

| 基準 | 由来 | 性質 |
|---|---|---|
| AMI | 特徴 | Adjusted Mutual Information |
| ARI | 特徴 | Adjusted Rand Index |
| V-Measure | 特徴 | Homogeneity + Completeness |
| FMI | 特徴 | Fowlkes-Mallows Index |
| BNM | 特徴 | Batch Nuclear-norm Maximization |
| RankMe | ロジット | 表現行列のランク（Garrido et al., 2023）|
| CHI | ロジット | Calinski-Harabasz Index |

7 基準で個別ランクづけ → **平均ランクが最低**のハイパラを選択。オラクル精度との絶対誤差 **2.6±4.3%**（単一基準: 2.8〜12.0%、Random: 8.9±8.1%）。

---

## V-PET vs MixMatch/FixMatch/FlexMatch の系譜上の対比

| 観点 | MixMatch (2019) | FixMatch (2020) | FlexMatch (2021) | **V-PET (2025)** |
|---|---|---|---|---|
| バックボーン | Wide ResNet（スクラッチ） | Wide ResNet（スクラッチ） | Wide ResNet（スクラッチ） | **VFM（CLIP/DINOv2）+ PEFT** |
| 疑似ラベル品質 | K=2 平均 + シャープニング | 弱拡張 + τ=0.95 閾値 | クラス別動的閾値 CPL | **VFM×PEFT アンサンブル投票** |
| ラベルなし損失 | Brier スコア | クロスエントロピー + 閾値 | 同上（クラス別閾値） | ソフトラベル on fine-tuning |
| 自己訓練 | 反復なし | 反復なし | 反復なし | **1 ラウンドのみ** |
| 信頼度閾値 | なし（ソフト） | **τ=0.95（高）** | **クラス別 T_t(c)** | **τ=0（全使用）** |
| 強い拡張 | RandAugment 程度 | RandAugment/CTAugment | RandAugment | 標準 fine-tune（不要） |
| MixUp | あり | なし | なし | なし |

V-PET の特徴：「**閾値を捨て、強拡張を捨て、複雑な損失設計を捨て、アンサンブルだけで品質を確保する**」というミニマリスト設計。

---

## アンサンブルの計算コスト

「N×M モデルを fine-tune」と聞くと重そうだが：
- 各 PEFT 訓練はラベルあきデータが少ないので軽い
- ステップ (a)〜(c) はステップ (d) より高速
- 全体実行時間は他 SSL ベースラインの **約 1.16 倍**

---

## 関連ページ

- [[sources/revisiting-ssl-foundation-models]]: 詳細な論文要約
- [[translations/revisiting-ssl-foundation-models]]: 日本語全文翻訳
- [[concepts/semi-supervised-learning]]: 半教師あり学習の全体像
- [[concepts/parameter-efficient-fine-tuning]]: PEFT の概念解説
- [[concepts/foundation-model]]: VFM が SSL/SeSL に持ち込んだパラダイム転換
- [[entities/clip]]: V-PET の主要バックボーン候補
- [[entities/dinov2]]: V-PET の主要バックボーン候補
- [[entities/fixmatch]]: 比較対象（VFM 上では性能低下）
- [[entities/flexmatch]]: 比較対象（同上）
