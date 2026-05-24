---
type: entity
entity_kind: model
aliases: [CLIP, OpenCLIP, Contrastive Language-Image Pre-training]
tags: [weakly-supervised, vision-language, foundation-model, openai]
related: [[concepts/weakly-supervised-pretraining]], [[concepts/foundation-model]], [[concepts/vision-transformer]]
sources: [[sources/dinov2-learning-robust-visual-features-without-supervision]]
updated: 2026-05-24
---

# CLIP / OpenCLIP

## 概要

**CLIP** = **Contrastive Language-Image Pre-training**。OpenAI の Radford ら（2021）が発表した、**画像 - テキスト対の対比学習**による事前学習法。CV における**汎用視覚基盤モデル**（[[concepts/foundation-model]]）の事実上の起点となった画期的論文。

- 論文: "Learning Transferable Visual Models From Natural Language Supervision"
- arXiv: 2103.00020 → ICML 2021
- OpenAI コード: <https://github.com/OpenAI/CLIP>

**OpenCLIP** は LAION コミュニティが公開データ（LAION-2B / LAION-5B）で CLIP を再現・拡張した版で、現在は実質的な「公開 CLIP」として広く使われている：

- コード: <https://github.com/mlfoundations/open_clip>
- 主要メンバー: Ilharco, Schuhmann ら

DINOv2 論文（[[sources/dinov2-learning-robust-visual-features-without-supervision]]）における**主要ベースライン**は OpenCLIP（特に ViT-G/14, LAION-2B 訓練）と EVA-CLIP。

## 学習設定（CLIP）

| 項目 | 値 |
|---|---|
| 学習データ | **WIT-400M**（Web から収集した 4 億の画像 - テキスト対、OpenAI 内部） |
| アーキテクチャ | 画像 encoder（ViT または ResNet）+ テキスト encoder（Transformer） |
| 損失 | **対比損失（InfoNCE）**: バッチ内のすべての対について「対応する画像とテキストの類似度を最大化、非対応を最小化」 |
| バッチサイズ | 32,768（巨大バッチが性能の鍵） |
| 公開モデル | ViT-B/16, ViT-B/32, ViT-L/14, ViT-L/14@336 |

### 対比損失の概略

バッチ内の N 個の画像 - テキスト対 $(I_i, T_i)$ について：
- 画像 encoder で $v_i = f_{img}(I_i)$、テキスト encoder で $t_i = f_{txt}(T_i)$ を得る
- $N \times N$ の類似度行列 $S_{ij} = v_i \cdot t_j / \|v_i\| \|t_j\|$ を計算
- 対角成分（対応する対）を大きく、それ以外を小さくするクロスエントロピー損失（行方向と列方向の両方）

> **補足: なぜ「対比」と呼ぶか** — 「正例（対応する対）」と「負例（同じバッチの他の画像 - テキスト組合せ）」を**対比**して学習する、という意味。詳細: [[concepts/weakly-supervised-pretraining]]

## CLIP の革命的だった点

1. **ゼロショット分類**: テキストプロンプト "a photo of a {class}" を作るだけで、未知のクラスにも対応可能
   - ImageNet で 76.2% top-1（学習データに ImageNet ラベルを 1 つも使わずに）
2. **共有埋め込み空間**: 画像とテキストが同じ空間にマップされ、テキスト → 画像検索、画像クラスタリング、Stable Diffusion のテキスト条件付けなどが容易に
3. **頑健性**: ImageNet 学習モデルが弱い ImageNet-R/A/Sketch でも、相対的に良く動く
4. **スケール則**: データとモデルを増やすと一貫して性能が上がる

## OpenCLIP の重要モデル

| Model | Arch | Data | ImageNet zero-shot | ImageNet linear |
|---|---|---|---|---|
| OpenCLIP ViT-B/32 | ViT-B/32 | LAION-400M | 62.9 | 76.6 |
| OpenCLIP ViT-L/14 | ViT-L/14 | LAION-2B | 75.5 | 84.4 |
| OpenCLIP ViT-H/14 | ViT-H/14 | LAION-2B | 78.0 | 84.4 |
| **OpenCLIP ViT-G/14** | ViT-G/14 | LAION-2B | 80.1 | **86.2** |
| **OpenCLIP ViT-bigG/14** | ViT-bigG/14 | LAION-2B | 82.0 | – |

DINOv2 が比較対象として最重要視するのは **OpenCLIP ViT-G/14**（86.2% linear ImageNet）。

## CLIP / OpenCLIP の弱点（DINOv2 論文が指摘）

[[sources/dinov2-learning-robust-visual-features-without-supervision]] §1, §7.4 の主張：

1. **キャプションは画像情報の近似でしかない**: 「a photo of a dog」というキャプションでは犬の品種・姿勢・背景・3D 構造の大半が落ちる。**特に密予測（segmentation, depth）で弱い**。
2. **テキスト - 画像対が必要**: 純粋画像のみのデータを使えない（DINOv2 の SSL アプローチの強み）
3. **計算コストが大きい**: テキスト encoder も並行訓練する。DINOv2 §9 の試算では **OpenCLIP-G の訓練は DINOv2-g の 10 倍の CO₂eq**。
4. **キャプションのバイアス**: Web キャプションは特定の表現分布を持ち、それを引き継ぐ

DINOv2 が dense prediction や fine-grained 分類で OpenCLIP を大きく上回るのは、この「キャプションが捉えきれないピクセル情報」を SSL が捉えられるため、という解釈。

## CLIP / OpenCLIP の派生

- **ALIGN**（Google, 2021）: 1.8B 対、CLIP より大規模
- **SLIP**（Mu et al., 2021）: CLIP + SSL
- **LiT**（Zhai et al., 2022）: テキスト encoder のみ訓練、画像 encoder は凍結
- **EVA-CLIP**（Sun et al., 2023）: より良いビジョン initialization、scale up
- **SigLIP**（Zhai et al., 2023）: softmax → sigmoid 損失で大バッチ不要に
- **MetaCLIP / DFN**: データキュレーションの改善
- **CLIPA**, **CoCa** など多数

## CLIP と VLM への発展

CLIP の image encoder は、その後の **VLM（Vision-Language Models）** の標準的な vision tower になった：

- **BLIP / BLIP-2**: CLIP encoder + Q-Former + LLM
- **LLaVA**: CLIP ViT-L/14 + LLaMA
- **LLaVA-NeXT**: CLIP + **DINOv2** の特徴連結（両者の良さを取る）
- **Flamingo, IDEFICS** など

DINOv2 vs CLIP は、現代 VLM の vision tower 選択における主要な議論軸となっている。

## 関連ページ

- [[sources/dinov2-learning-robust-visual-features-without-supervision]]: CLIP / OpenCLIP を主要ベースラインとして比較する論文
- [[concepts/weakly-supervised-pretraining]]: CLIP が代表する系統の解説
- [[concepts/foundation-model]]: CLIP は CV 初の本格的基盤モデル
- [[entities/dinov2]]: CLIP に対する純粋 SSL 側からの対抗
