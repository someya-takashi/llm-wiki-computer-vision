---
type: entity
entity_kind: model
aliases: [InceptionNeXt, Inception depthwise convolution, MetaNeXt, InceptionNeXt-A, InceptionNeXt-T, InceptionNeXt-S, InceptionNeXt-B]
tags: [inceptionnext, convnext, metaformer, depthwise-convolution, throughput, efficiency, convolutional-neural-network, backbone]
related: ["[[entities/convnext]]", "[[entities/rdnet]]", "[[concepts/convolutional-neural-network]]", "[[concepts/skip-connection]]", "[[entities/swin-transformer]]", "[[entities/nfnet]]"]
sources: ["[[sources/inceptionnext]]"]
updated: 2026-09-01
---

# InceptionNeXt

Weihao Yu, Pan Zhou, Shuicheng Yan, **Xinchao Wang**（NUS / SMU / Sea AI Lab / Skywork AI）による CNN バックボーン（arXiv:2303.16900, **CVPR 2024**）。詳細解説: [[sources/inceptionnext]] / 翻訳: [[translations/inceptionnext]]。

## 一言で

**[[entities/convnext]] の 7×7 depthwise 畳み込みを 4 分岐（3×3 + 1×11 + 11×1 + 恒等写像）に分解し、訓練スループット 1.6 倍で精度 +0.2% を得たモデル。** 出発点は「ConvNeXt-T は ResNet-50 と同 FLOPs なのに A100 のスループットは約 60% しかない」という実測。

## 中核: Inception depthwise convolution

入力をチャネル方向に 4 分割し、それぞれ別処理して連結する。

| 分岐 | 処理 | チャネル数（既定） |
|---|---|---|
| $X_{\mathrm{hw}}$ | `DWConv 3×3` | $C/8$ |
| $X_{\mathrm{w}}$ | `DWConv 1×11` | $C/8$ |
| $X_{\mathrm{h}}$ | `DWConv 11×1` | $C/8$ |
| $X_{\mathrm{id}}$ | **恒等写像** | $5C/8$ |

**畳み込みを通るのは全チャネルの 3/8 だけ。** 残り 5/8 はそのまま通り抜ける（ShuffleNet V2 の「全チャネルを畳み込みに通す必要はない」という知見）。

```python
class InceptionDWConv2d(nn.Module):
    def __init__(self, in_channels, square_kernel_size=3,
                 band_kernel_size=11, branch_ratio=1/8):
        gc = int(in_channels * branch_ratio)
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size,
                                   padding=square_kernel_size//2, groups=gc)
        self.dwconv_w  = nn.Conv2d(gc, gc, (1, band_kernel_size),
                                   padding=(0, band_kernel_size//2), groups=gc)
        self.dwconv_h  = nn.Conv2d(gc, gc, (band_kernel_size, 1),
                                   padding=(band_kernel_size//2, 0), groups=gc)
        self.split_indexes = (gc, gc, gc, in_channels - 3 * gc)

    def forward(self, x):
        x_hw, x_w, x_h, x_id = torch.split(x, self.split_indexes, dim=1)
        return torch.cat((self.dwconv_hw(x_hw), self.dwconv_w(x_w),
                          self.dwconv_h(x_h), x_id), dim=1)
```

**計算量が $k$ の 2 次から 1 次に落ちる**のが本質。

| 畳み込みの型 | Params | FLOPs |
|---|---|---|
| 通常の畳み込み | $k^{2}C^{2}$ | $2k^{2}C^{2}HW$ |
| Depthwise 畳み込み | $k^{2}C$ | $2k^{2}CHW$ |
| **Inception depthwise 畳み込み** | $(2k+9)C/8$ | $(2k+9)CHW/4$ |

## MetaNeXt という抽象化

**MetaFormer**（同じ筆頭著者 Weihao Yu の先行研究、本 wiki 未取り込み ⚠）の枠組みで ConvNeXt を抽象化したもの。

| | MetaFormer | **MetaNeXt** |
|---|---|---|
| ショートカット | **2 本**（トークン混合器用・MLP 用） | **1 本**（統合） |
| 構造 | Norm → Mixer → (+) → Norm → MLP → (+) | Mixer → Norm → MLP → (+) |
| トークン混合器の制約 | 自己注意でも可 | **複雑すぎると収束しない** |

ConvNeXt = MetaNeXt の Mixer を `DWConv 7×7` にしたもの。InceptionNeXt = Inception depthwise 畳み込みにしたもの。

**MetaNeXt-Attn（Mixer を自己注意にした版）は top-1 3.9% で収束しない**（表5）。ショートカットを 1 本に減らした代償で、[[concepts/skip-connection]] の主題の直接的な実証になっている。

## バリアント

**表**: InceptionNeXt の構成（原典 表3）と ImageNet-1K 結果（表4）

| | **A**（Atto） | **T** | **S** | **B** |
|---|---|---|---|---|
| 埋め込み次元 | 40/90/180/320 | 96/192/384/768 | 96/192/384/768 | 128/256/512/1024 |
| ブロック数 | 2/2/6/2 | 3/3/9/3 | 3/3/27/3 | 3/3/27/3 |
| **帯状カーネル** | **9** | 11 | 11 | 11 |
| **畳み込み分岐比** | **1/4** | 1/8 | 1/8 | 1/8 |
| MLP 比 | 4/4/4/**3** | 4/4/4/**3** | 4/4/4/**3** | 4/4/4/**3** |
| Params (M) | 4.2 | 28 | — | — |
| MACs (G) | 0.51 | 4.2 | — | — |
| **A100 訓練 (img/s)** | **2661** | **901** | — | — |
| A100 推論 (img/s) | 9876 | 2900 | — | — |
| Top-1 (%) | 75.3 | **82.3** | — | — |

- **正規化は BatchNorm**（ConvNeXt は LayerNorm）。LN の方が精度は +0.1 高い（82.4）が**訓練スループットが 20% 落ちる**ため、速度優先で BN を選択
- **第 4 ステージだけ MLP 比 3**（他は 4）。節約分を分類器に回す
- 訓練: A は 450 エポック / bs 1280 / lr 1e-3、T・S・B は 300 エポック / bs 4096 / lr 4e-3。384² ファインチューンは 30 エポック / EMA 0.9999

## ConvNeXt との比較

**表**: ImageNet-1K（原典 表4）

| Model | Params | MACs | A100 訓練 | A100 推論 | Top-1 |
|---|---|---|---|---|---|
| ConvNeXt-A | 3.7 | 0.55 | 835 | 4539 | 75.7 |
| **InceptionNeXt-A** | 4.2 | 0.51 | **2661 (+219%)** | **9876 (+118%)** | 75.3 (−0.4) |
| ConvNeXt-T | 29 | 4.5 | 575 | 2413 | 82.1 |
| **InceptionNeXt-T** | 28 | 4.2 | **901 (+57%)** | **2900 (+20%)** | **82.3 (+0.2)** |
| ResNet-50（参考） | 26 | 4.1 | 969 | 3149 | 78.4 |
| [[entities/swin-transformer\|Swin-T]]（参考） | 29 | 4.5 | 564 | 1768 | 81.3 |

**軽量ほど効き、大きくなるほど効かない。** 論文の説明は明快で、**depthwise 部分の計算量が $\mathcal{O}(C)$、MLP が $\mathcal{O}(C^2)$** なので、$C$ が大きくなると MLP が支配し depthwise を速くしても全体には効かない。Atto +219% → Tiny +57% と減衰する。

## セマンティックセグメンテーション

**表**: ADE20K（原典 表6・表7）

| Backbone | Head | Params | FPS | mIoU |
|---|---|---|---|---|
| Swin-T | UperNet | 60 | 20.6 | 45.8 |
| ConvNeXt-T | UperNet | 60 | 20.6 | 46.7 |
| **InceptionNeXt-T** | UperNet | **56** | **22.7** | **47.9** |
| **InceptionNeXt-S** | UperNet | 78 | 17.6 | **50.0** |
| **InceptionNeXt-B** | UperNet | 115 | 17.5 | **50.6** |
| PoolFormer-S24 | FPN | 23 | 28.8 | 40.3 |
| **InceptionNeXt-T** | FPN | 28 | **31.4** | **43.1** |

**パラメータが少なく FPS が高く mIoU も高い**という三方良し。UperNet の stochastic depth は T/S/B = 0.2/0.3/0.4、FPN は 0.1/0.2/0.2。

## アブレーションから読める設計の勘所

| 変更 | 訓練 tput | Top-1 |
|---|---|---|
| ベースライン（InceptionNeXt-T） | 901 | **82.3** |
| **帯状カーネル（水平 or 垂直）を除去** | 947 / 954 | **81.9（−0.4、最大の劣化）** |
| 3×3 の正方カーネルを除去 | 940 | 82.0（−0.3。**速度優先なら落としてよい**） |
| 帯状カーネル 11 → 13 | 896 | 82.0（**大きくすると下がる**） |
| 分岐比 1/8 → 1/16 | 936 | 81.8 |
| BatchNorm → LayerNorm | **721（−20%）** | 82.4 |

**帯状カーネルの 2 本が本体**で、これが受容野を広げている。**11 で頭打ちで 13 では下がる**（論文は「最適化に起因するかもしれない」と留保）。

## 系譜・位置づけ

- **[[entities/convnext]] への実測ベースの批判**。ConvNeXt が「3 → 5 → 7 で改善し 7×7 で飽和」と結論した近代化ロードマップに対し、**その 7×7 が実機で払っているコスト**を ResNet-50 という同 FLOPs のベースラインで可視化した。
- **[[entities/rdnet]] と同じ診断・別の処方**。両者とも 2024 年、両者とも「ConvNeXt の 7×7 depthwise が遅い」から出発するが、InceptionNeXt は**カーネルの分解**、RDNet は**ショートカットの連結化**で応じた。**RDNet の比較表では InceptionNeXt-T が 132ms と表中最速**、RDNet-T は 175ms で 82.8。**速度で InceptionNeXt、精度で RDNet。**
- **[[entities/nfnet]] と態度が同型**。NFNet が「理論 FLOPS でも推論レイテンシでもなく実機の訓練レイテンシを最適化対象にした」のと同じく、本論文も**訓練スループットを一級の設計目標**に置いている。
- **[[concepts/skip-connection]] に一次情報を追加**。MetaNeXt がショートカットを 2 本→1 本にした結果、自己注意を Mixer に置くと収束しなくなる（3.9%）。**ショートカットの本数が他の部品の選択肢を制限する。**
- **精度の絶対値では最新モデルに届いていない**（82.3 は FocalNet-T と同点、RDNet-T 82.8 に負ける）。本モデルの価値は**精度あたりの速度**にある。論文自身が「炭素排出量を削減するための**経済的なベースライン**」と位置づけている。

## 弱点

- **大きなモデルほど効果が薄い**（$\mathcal{O}(C)$ vs $\mathcal{O}(C^2)$）。エッジ・軽量向けの手法
- **帯状カーネル 13 で性能が落ちる理由が未解明**
- **MetaNeXt の適用範囲が狭い**（自己注意で破綻）
- **BatchNorm を選んだ副作用が未検証**。[[entities/nfnet]] が整理した BN の既知の問題（バッチ依存、対比学習での情報漏洩）を踏まえると、**下流の自己教師あり学習への適性**は本論文の射程外
- **スループット測定の条件依存**。バッチサイズは「GPU が収容できるまで減らす」可変設定で、1 条件あたり 1 つの数字しか出ていない

## 公開

- コード / 重み: <https://github.com/sail-sg/inceptionnext>
- `timm` に `inception_next_atto` / `_tiny` / `_small` / `_base` として収録

## 関連ページ

- [[sources/inceptionnext]] — 論文要約（最重要）
- [[translations/inceptionnext]] — 全文和訳（Appendix A・B 込み）
- [[entities/convnext]] / [[sources/convnext]] — 直接の批判対象
- [[entities/rdnet]] / [[sources/rdnet]] — 同じ診断・別の処方
- [[concepts/skip-connection]] — MetaNeXt のショートカット統合とその代償
- [[concepts/convolutional-neural-network]] — カーネルサイズと depthwise 畳み込みの議論
- [[entities/nfnet]] — 実機レイテンシを最適化対象にする同型の態度
- [[entities/swin-transformer]] — スループットの比較対象
