---
type: entity
entity_kind: model
aliases: [ConvNeXt V2, ConvNeXtV2, FCMAE, GRN]
tags: [convnext-v2, fcmae, grn, convolutional-neural-network, masked-image-modeling, self-supervised, fair, meta-ai]
related: ["[[entities/convnext]]", "[[concepts/convolutional-neural-network]]", "[[concepts/masked-image-modeling]]", "[[entities/mae]]", "[[entities/hiera]]", "[[entities/maxvit]]"]
sources: ["[[sources/convnext-v2]]"]
updated: 2026-06-17
---

# ConvNeXt V2

Meta AI（FAIR）+ NYU + KAIST の **Sanghyun Woo, Zhuang Liu, Saining Xie ら**による、自己教師あり学習と共設計された ConvNet ファミリー（arXiv:2301.00808, **CVPR 2023**）。詳細解説: [[sources/convnext-v2]] / 翻訳: [[translations/convnext-v2]]。

## 一言で

**[[entities/convnext|ConvNeXt V1]] のブロックに GRN 層を 1 つ足し、疎畳み込みベースの MAE（FCMAE）で事前学習した**もの。V1 が抱えていた「**CNN は MIM と相性が悪い**」という制約を解消し、ImageNet で **88.9%**（公開データのみで当時 SOTA）を達成した。

## 2 つの構成要素

### FCMAE（Fully Convolutional Masked AutoEncoder）— 事前学習の枠組み

- **マスク画像を「2D の疎な画素配列」とみなし、疎畳み込み（submanifold sparse convolution）で可視部分のみ処理**。これによりマスク領域からの情報漏洩（ショートカット学習）を遮断する。**この 1 点で 79.3 → 83.7 と +4.4**
- ファインチューニング時は**特別な扱いなしに標準の密畳み込みへ戻せる**
- **マスク率 0.6**、マスク単位 32×32。最終段階でマスクを生成し再帰的にアップサンプリング
- **デコーダは単一の ConvNeXt ブロック**（次元 512）。UNet 比 **1.7× 高速**で精度は同等
- 損失はマスク領域のみのパッチ正規化 MSE（[[entities/mae]] と同じ）
- 副産物として**スループット 1.3×、最大メモリ 1/2**

### GRN（Global Response Normalization）— アーキテクチャの変更

チャネル間に競合を持ち込み、FCMAE 訓練時に起きる**特徴崩壊**（チャネル間で活性化が冗長になる現象）を防ぐ。

$$X_{i}=\gamma*X_{i}*\mathcal{N}(\mathcal{G}(X)_{i})+\beta+X_{i}$$

1. **集約** $\mathcal{G}$: 各チャネルを空間方向の **L2 ノルム**でスカラー化
2. **正規化** $\mathcal{N}$: **除法正規化** $||X_i|| / \sum_j ||X_j||$（他チャネルとの相対的重要度）
3. **較正**: 元の応答に掛け戻す

- $\gamma, \beta$ は**ゼロ初期化** + 残差接続 → 最初は恒等写像として振る舞い、訓練とともに効いてくる
- **実装 3 行、追加パラメータ・FLOPs は実質ゼロ**
- 大域平均プーリング（83.7）ではなく **L2 ノルム（84.6）** でなければならない
- **BN は 80.5 と大きく落ちる**（バッチ軸の空間正規化がマスク入力と相性が悪い）
- SE / CBAM でも 84.4-84.5 は出るが、GRN は**パラメータ増なしで同等以上**（89M vs 109M）

## ブロック構造（V1 からの差分）

```
入力 (C ch)
  ├─ d7×7, C          ← V1 と同じ
  ├─ LayerNorm        ← V1 と同じ
  ├─ 1×1, C → 4C      ← V1 と同じ
  ├─ GELU             ← V1 と同じ
  ├─ GRN              ← ★ V2 で追加
  ├─ 1×1, 4C → C      ← V1 と同じ
  └─ + 残差接続        ← ★ LayerScale を削除（GRN により冗長化）
```

**変更は「GRN を足して LayerScale を消す」だけ**。stem・stage 構成・カーネルサイズ等は [[entities/convnext]] と完全に同じ。

## バリアント（8 サイズ）

**表**: ConvNeXt V2 の構成と ImageNet 性能（[[sources/convnext-v2]] Appendix A.1 / 表14 / 表15 より）

| モデル | $C$ | $B$ | params | FLOPs | V1 教師あり | **V2 + FCMAE** | IN-22K 中間 FT |
|---|---|---|---|---|---|---|---|
| Atto (A) | 40 | (2,2,6,2) | 3.7M | 0.55G | 75.7 | **76.7** (+1.0) | — |
| Femto (F) | 48 | (2,2,6,2) | 5.2M | 0.78G | 77.5 | **78.5** (+1.0) | — |
| Pico (P) | 64 | (2,2,6,2) | 9.1M | 1.37G | 79.5 | **80.3** (+0.8) | — |
| Nano (N) | 80 | (2,2,8,2) | 15.6M | 2.45G | 80.8 | **81.9** (+1.1) | 82.1 / 83.4 (384²) |
| Tiny (T) | 96 | (3,3,9,3) | 28.6M | 4.47G | 82.1 | **83.0** (+0.9) | 83.9 / 85.1 (384²) |
| Base (B) | 128 | (3,3,27,3) | 89M | 15.4G | 83.8 | **84.9** (+1.1) | 86.8 / 87.7 (384²) |
| Large (L) | 192 | (3,3,27,3) | 198M | 34.4G | 84.3 | **85.8** (+1.5) | 87.3 / 88.2 (384²) |
| Huge (H) | 352 | (3,3,27,3) | 659M | 115G | — | **86.3** | 88.7 (384²) / **88.9** (512²) |

- **V1 との対応**: T/B/L は V1 と同一構成。**A/F/P/N は V1 になかった超軽量帯**（別研究由来）、**H は V2 で新設**（V1 の XL = $C$=256 は V2 に存在しない）
- **V2-Base が V1-Large を、V2-Large が V1-XLarge を上回る**（IN-22K 中間 FT: 86.8/87.7 vs 86.6/87.5、87.3/88.2 vs 87.0/87.8）。共設計だけで 1 段階上のサイズ相当の性能が得られる

## 主要結果

- **ImageNet-1K**: ConvNeXt V2-H @512² で **88.9%**（IN-22K 中間 FT、**公開データのみで当時 SOTA**）。MViTV2-H 88.8 / **[[entities/maxvit|MaxViT]]-XL 88.7** / CoAtNet-4 88.1 を上回る
- **COCO**（Mask R-CNN）: V2-H で **55.7 AP^box / 48.9 AP^mask**（Swin V2-H SimMIM 54.4）
- **ADE20K**（UPerNet）: V2-H で **55.0 mIoU**、IN-22K FT + 640² で **57.0**（Swin V2-H SimMIM 54.2）
- **他 MIM との比較**: SimMIM（Swin）は全サイズで上回る。MAE（ViT）は Large まで互角（**198M で 85.8 vs ViT-L 307M で 85.9**）だが、**Huge では 86.3 vs ViT-H 86.9 と負ける**
- **MoCo v3 との比較**: 同じ V2-B で FCMAE 84.9 > 教師あり 300ep 84.3 > MoCo v3 83.7。**MIM > 対比学習という ViT 側の知見が ConvNet でも成立**

## 系譜・位置づけ

- **直接の前作は [[entities/convnext]]**。V1 が教師あり学習の土俵で「ConvNet は Swin と互角」を示したのに対し、V2 は**自己教師あり学習の土俵に持ち込んだ**。
- **[[entities/mae]] の直接の子孫**。非対称エンコーダ・デコーダとパッチ正規化 MSE を継承しつつ、「エンコーダが可視部分だけ処理する」を疎畳み込みで実現した。
- **[[entities/hiera]] と同じ発想の ConvNet 版**。Hiera が「MAE 互換であること」を設計目標に階層型 ViT を作り直したのと同様、V2 は FCMAE 互換であることを設計目標に ConvNeXt を修正した。両者とも **co-design** の実例。
- **本 wiki が V1 のページに書いた「ConvNeXt は MIM と相性が悪い」という制約に対する回答**にあたる。ただし解決したのは MIM 適性のみで、**言語との接続性という ViT のもう一つの優位はそのまま**であり、その後も CLIP / MLLM の視覚エンコーダは ViT が主流であり続けた。
- **[[entities/dinov3]] が蒸留先として提供する ConvNeXt は V1 系**（ConvNeXt-T/S/B/L, 28M/50M/88M/198M）であり、V2 ではない点に注意。

## 公開

- コード / 重み: <https://github.com/facebookresearch/ConvNeXt-V2>
- 本文の実験は TPU v3-256 pod（JAX、密マスク実装）。**PyTorch 再現実装も公開**（GPU では MinkowskiEngine による疎畳み込み）
- `timm` に `convnextv2_atto` 〜 `convnextv2_huge` として収録

## 関連ページ

- [[sources/convnext-v2]] — 論文要約（最重要）
- [[translations/convnext-v2]] — 全文和訳（Appendix A〜D 込み）
- [[entities/convnext]] / [[sources/convnext]] — 直接の前作
- [[entities/mae]] / [[sources/mae]] — FCMAE の下敷き
- [[concepts/masked-image-modeling]] — 本モデルが「CNN でも MIM は効く」を示した領域
- [[concepts/convolutional-neural-network]] — CNN の帰納バイアスと部品
- [[entities/hiera]] — 同じ co-design 発想の Transformer 側
