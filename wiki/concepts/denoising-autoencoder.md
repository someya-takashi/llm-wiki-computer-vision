---
type: concept
aliases: [DAE, Denoising Autoencoder, ノイズ除去オートエンコーダ]
tags: [paradigm, representation-learning, foundational]
related: [[self-supervised-learning]], [[masked-image-modeling]]
sources: [[sources/mae]]
updated: 2026-05-25
---

# Denoising Autoencoder（DAE, ノイズ除去オートエンコーダ）

## 一言で

**入力信号にノイズや破損を加え、元の破損していない信号を再構成することを学習する**オートエンコーダの一形態。Vincent ら（2008, ICML）が提案し、深層学習における自己教師あり表現学習の最も古典的な枠組みの 1 つ。**現代の Masked Image Modeling（MIM, [[masked-image-modeling]]）、MAE（[[entities/mae]]）、BERT の MLM などはすべて DAE の特殊形**として理解できる。

---

## なぜ重要か

DAE は深層学習表現学習の**理論的基礎**を提供している：

1. **自己教師あり学習の最も古い形式の 1 つ**: ラベル不要で、データ自体の構造から学習信号を作る
2. **「破損 → 再構成」というシンプルな枠組みが現代手法すべてに通底する**: MIM, MAE, BERT MLM, GPT, 拡散モデルまで DAE の枠組みで理解できる
3. **「冗長性を抜いて残りから復元させると有用な表現が学べる」という洞察**: 後の SSL 全般の中核アイデア

---

## 基本構造

```
[元の信号 x]
   ↓ 破損（noise / mask / removal）
[破損信号 x̃]
   ↓ encoder f_θ
[潜在表現 h]
   ↓ decoder g_φ
[再構成 x̂]
   ↓
損失 L(x, x̂)
```

学習目的:

$$
\min_{\theta, \phi} \mathbb{E}_{x \sim D, \tilde{x} \sim C(\cdot|x)} \big[ L(x, g_\phi(f_\theta(\tilde{x}))) \big]
$$

- $D$: データ分布
- $C(\cdot|x)$: 破損プロセス（noise の種類）
- $L$: 再構成損失（通常 MSE またはクロスエントロピー）

---

## 破損プロセスの種類

DAE は「**どんな破損を加えるか**」で多様な変種を持つ：

| 破損の種類 | 代表手法 |
|---|---|
| **Gaussian noise を加える** | 元の DAE（Vincent 2008） |
| **入力次元の一部をランダムに 0 にする（masking noise）** | 元の DAE の派生、現代の MIM の祖先 |
| **ピクセルを欠落させる** | Context Encoder (Pathak 2016) → MAE |
| **色チャネルを除去する** | Colorization (Zhang 2016) |
| **画像内のパッチを並び替える** | Jigsaw Puzzle (Noroozi 2016) |
| **時系列の部分を隠す（言語の単語マスク）** | BERT の MLM |
| **入力を低解像度化する** | Super-Resolution as SSL |
| **連続的に noise を加えて段階的に復元する** | 拡散モデル（DDPM 等）— 一般化された DAE |

---

## MAE と DAE の関係

MAE 論文（[[sources/mae]]）の §2 で、著者ら自身が「**MAE はノイズ除去オートエンコーディングの一形態**」と明確に位置づける：

> Our MAE is a form of denoising autoencoding, but different from the classical DAE in numerous ways.

具体的に MAE が DAE と違う点：

| | 古典的 DAE | MAE |
|---|---|---|
| アーキテクチャ | 対称（encoder と decoder が同サイズ）| **非対称**（軽量 decoder）|
| 破損 | Gaussian noise や少量のマスキング | **75% という極端な masking** |
| 入力処理 | 破損信号を全部 encoder に通す | **可視部分のみ encoder、マスクは decoder で扱う** |
| ターゲット | 全画素 | **マスク位置のみ**で損失計算 |
| バックボーン | MLP / CNN | **ViT** |

つまり MAE は「DAE という古典的枠組みを Transformer 時代に再発明し、いくつかの重要な設計判断（非対称性、高マスク率、可視のみ encoder）で大規模化に成功した」と理解できる。

---

## 歴史的系譜

```
[1980s] Hopfield Networks (auto-associative memory)
   ↓
[1990s] 古典的 Autoencoder（圧縮・PCA との関係）
   ↓
[2008] Vincent et al. "Denoising Autoencoders" (ICML)
   ↓
[2010] Stacked Denoising Autoencoders (JMLR)
   ↓
[2010s 前半] CV では一旦下火（教師あり学習が席巻）
   ↓
[2016] Context Encoder (画像 inpainting として復活)
   ↓
[2018] BERT — NLP で MLM として爆発
   ↓
[2020-2021] iGPT, ViT MJP, BEiT — CV で MIM として再挑戦
   ↓
[2022] MAE — シンプルでスケーラブルな実装
   ↓
[2022-2025] iBOT / DINOv2 / DINOv3 — MIM を識別型 SSL と統合
   ↓
[2020s] 拡散モデル — DAE の枠組みを連続ノイズ過程に拡張
```

---

## DAE 視点で見る現代の SSL

意外なことに、現代の主要 SSL 手法はほぼすべて DAE の枠組みで再解釈できる：

| 手法 | 破損 | 再構成ターゲット |
|---|---|---|
| BERT MLM | 単語の 15% を `[MASK]` | 元の単語 |
| GPT | 系列の suffix を隠す | 次のトークン |
| **MAE** | **画素 75% をマスク** | **元の画素** |
| BEiT | 画素を一部マスク | dVAE 離散トークン |
| iBOT | パッチを一部マスク | teacher の特徴量 |
| Colorization | 色を除去 | 元の色 |
| Inpainting | 矩形領域を除去 | 元の画素 |
| 拡散モデル | 連続的 Gaussian noise | 元の画像（または noise） |
| SimCLR / DINO（対比学習） | 異なる augmentation で「破損」 | 元画像と一致する表現 |

最後の対比学習は厳密には DAE と異なるが、「**augmentation という破損から元の画像を識別する**」と読み替えれば類似性が見える。

---

## 拡散モデルとの繋がり

**拡散モデル（DDPM, Stable Diffusion 等）は DAE の連続版**として理解できる：

- DAE: **離散的な単一ステップ**で破損 → 復元
- 拡散モデル: **連続的に T ステップにわたって徐々に Gaussian noise を加え、各ステップで逆向きにノイズを除去**

実際 DDPM の損失は「**各時刻 t で加えられた noise を予測する**」形であり、本質的に DAE の連続版 + 段階的近似。

このため「**MAE と拡散モデルは祖先を共有する**」という見方ができ、両者の統合や相互応用も研究されている。

---

## 関連ページ

- [[sources/mae]] — DAE 視点で MAE を位置づける論文
- [[entities/mae]] — DAE の現代的成功例
- [[concepts/masked-image-modeling]] — MIM = DAE の masking 変種
- [[concepts/self-supervised-learning]] — DAE は SSL の最古の形式の 1 つ
- [[concepts/diffusion-model]] — DDPM / Stable Diffusion 等。DAE の連続版として位置づけられる生成モデル
- [[sources/i-synmed]] — DDPM 合成データで DINO を事前学習する医療応用例（IEEE Access 2025）
