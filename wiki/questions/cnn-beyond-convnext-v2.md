---
type: question
asked: 2026-06-17
question: "CNN で ConvNeXt V2 の性能を超えるモデルはあるでしょうか？"
sources_used: ["[[sources/convnext-v2]]", "[[sources/convnext]]", "[[sources/maxvit]]", "[[translations/maxvit]]", "[[entities/convnext-v2]]", "[[concepts/convolutional-neural-network]]", "[[entities/dinov3]]", "[[sources/nfnet]]", "[[entities/nfnet]]"]
---

# CNN で ConvNeXt V2 を超えるモデルはあるか

## 結論を先に

**あります。** ただし「何を CNN と数えるか」「データ条件を揃えるか」で答えが変わります。

| 問い | 答え |
|---|---|
| ConvNeXt V2 を超える CNN はあるか | **ある**（NFNet-F4+ 89.20 / InternImage-H 89.6） |
| **同じデータ条件で**超えているか | **いいえ**。いずれも 20〜30 倍のデータを使用 |
| 純粋な**静的カーネル**の CNN に限れば | **ConvNeXt V2 は依然として上位** |
| そもそも今も重要な問いか | **ImageNet 分類の SOTA 競争自体が主戦場から外れた** |

---

## 1. 比較の基準点を正確に置く

**ConvNeXt V2-H = 88.9%**（[[sources/convnext-v2]]）。この数字には条件が付いています。

- **IN-1K + IN-22K のみ**（約 1400 万枚）＝ **公開データのみ**
- 659M パラメータ、512²、IN-22K 中間ファインチューニング
- FCMAE 事前学習 + GRN（詳細: [[questions/convnext-v1-vs-v2-architecture]]）

論文自身の主張も「**公開データのみを使う手法として** SOTA」であって、無条件の SOTA ではありません。**この但し書きが本ページの議論の起点**になります。

---

## 2. wiki 内に既にある反例: NFNet-F4+

[[translations/maxvit]] の表 3 / 表 14 に、ConvNeXt V2 より高い純粋 CNN が載っています。

**表**: 大規模事前学習での上位モデル（[[sources/maxvit]] 表3 より。● = 純粋 ConvNet、⋄ = ハイブリッド）

| モデル | 種別 | params | IN-1K top-1 | 事前学習データ |
|---|---|---|---|---|
| ● EfficientNetV2-XL | CNN | 208M | 87.3 | IN-21K（公開） |
| ● ConvNeXt-XL（V1） | CNN | 350M | 87.8 | IN-21K（公開） |
| **● ConvNeXt V2-H** | **CNN** | 659M | **88.9** | **IN-22K（公開）** |
| **● NFNet-F4+** | **CNN** | 527M | **89.20** | **JFT-300M（非公開）** |
| ⋄ MaxViT-XL | ハイブリッド | 475M | 89.53 | JFT-300M |
| ⋄ CoAtNet-5 | ハイブリッド | 688M | 89.77 | JFT-300M |

**[[entities/nfnet|NFNet]]**（Normalizer-Free Network, Brock et al., DeepMind, ICML 2021、[[sources/nfnet]]）は **BatchNorm を完全に排した純粋 CNN** で、JFT-300M を使えば **89.2%** と ConvNeXt V2 を上回ります。**AGC（勾配ノルム / 重みノルムの比でクリップ）** と Scaled Weight Standardization で BN の役割を代替し、追加データなしでも **86.5%**（当時 SOTA）、**EfficientNet-B7 と同精度を 8.7× 速い訓練で**達成しました。

> **ただしこれは同じ土俵ではない** — JFT-300M は Google 内部の 3 億枚データセットで、IN-22K（1400 万枚）の**約 21 倍**。ConvNeXt V2 は JFT で訓練していないため、この 2 つを並べて「NFNet の方が強い」とは言えません。ConvNeXt V2 が「公開データのみで」と限定したのはまさにこの理由からです。

---

## 3. wiki 外の本命: InternImage

**データ条件を含めて考えても ConvNeXt V2 を上回る CNN** として最も有力なのが **InternImage**（Wang et al., CVPR 2023, Shanghai AI Lab）です。

> ⚠ **本 wiki 未取り込み**。以下は一次情報（ingest 済み原典）に基づかない記述であり、他セクションより確度が落ちます。数値を引用する際は原典で確認してください。

- **DCNv3（deformable convolution v3）** を中核演算子とする CNN
- **InternImage-H で IN-1K 89.6%**、COCO test-dev 65.4 AP（当時 SOTA）
- 事前学習は**公開データ**（Laion-400M / YFCC15M / CC12M 等、計 4 億枚規模）

「公開データのみ」という ConvNeXt V2 と同じ条件では上回りますが、**データ量が約 30 倍**（1400 万 vs 4 億枚）なので、これもアーキテクチャの優劣を直接示す比較にはなりません。

なお InternImage は OpenGVLab の Intern シリーズ（InternImage / InternVideo / InternVL / InternLM）の一角であり、本 wiki に既にある [[entities/internvl]] 系列と**同じ研究室の産物**です。

---

## 4. 最大の論点: 「それは本当に CNN か」

ここが本質的に面白い部分です。

**DCNv3 は入力に応じてサンプリング位置（オフセット）を学習します**。つまり畳み込みでありながら**内容依存の適応性**を持つ——これは attention の核心的性質そのものです。InternImage 論文自身が「**long-range dependence と adaptive spatial aggregation を畳み込みに持ち込む**」と説明しています。

これは本 wiki が繰り返し記録してきた「**相手の性質を輸入する**」という構図と同じです。

| 研究 | 何を輸入したか |
|---|---|
| [[entities/swin-transformer]]（2021） | Transformer ← **CNN の階層性・局所性** |
| [[entities/convnext]]（2022） | CNN ← **Transformer の訓練レシピ・設計** |
| [[entities/maxvit]]（2022） | 両者を 1 ブロックに**統合** |
| [[entities/convnext-v2]]（2023） | CNN ← **MIM（自己教師あり）との互換性** |
| **InternImage**（2023） | CNN ← **attention の内容依存的な適応性** |

つまり「純粋 CNN が ConvNeXt V2 を超えた」というより、**「attention の性質を借りた畳み込みが超えた」**と読むのが正確です。

> **静的カーネルという線引きで見ると** — カーネルの重みと位置が入力に依存しない古典的な意味での CNN（VGG / ResNet / EfficientNet / ConvNeXt 系）に限れば、**ConvNeXt V2 の 88.9% は今も上位に残ります**。NFNet も静的カーネルですが、勝っているのはデータ量の差によるものです。

---

## 5. より重要なこと: この問い自体が主戦場から外れた

2023 年以降、**ImageNet 分類の top-1 を競うこと自体が研究の主戦場ではなくなりました**。[[concepts/convolutional-neural-network]] の「CNN は終わったのか」節で整理した通り、評価軸は次に移っています。

| 現在の主要な評価軸 | CNN 系の採用状況 |
|---|---|
| **凍結特徴量の質**（[[entities/dinov2]] / [[entities/dinov3]]） | ほぼなし（ViT が主流） |
| **言語との整合性**（[[entities/clip]] / [[entities/siglip]]） | ほぼなし |
| **MLLM の視覚エンコーダ**（[[entities/qwen3-vl]] / [[entities/internvl-3]] 等） | ほぼなし |
| **効率・エッジ展開** | **CNN が現役**（下記） |

これらの軸で CNN が選ばれない理由は**精度ではなく、パッチ列表現が言語トークンと素直に接続できるかどうか**です。ConvNeXt V2 が MIM 非互換という弱点を解消してもなお、この軸は手つかずのままでした。

**CNN が現在も明確に選ばれている場所は「効率が要る領域」**です。代表例が [[entities/dinov3]]（2025）で、**ViT-7B から ConvNeXt-T/S/B/L（V1 系）を蒸留**して量子化・エッジ展開向けに配布しています。**性能で競うのではなく、性能を蒸留して受け取る側**に回った、というのが現在地です。

---

## 6. 公平に比較するには何を揃えるべきか

本ページの議論から抽出できる、CNN vs ViT の比較を読むときのチェックリスト。

1. **事前学習データの規模と公開性** — IN-1K（120 万）/ IN-22K（1400 万）/ JFT-300M（3 億、非公開）/ LAION 系（4 億〜）。**1 桁違えば結論は変わる**
2. **パラメータ数と FLOPs** — [[sources/convnext]] が「同 FLOPs で比較する」作法を確立し、[[sources/maxvit]] の「grid を block に置換（同パラメータ・同 FLOPs）」がその最も鋭い応用例
3. **訓練レシピ** — [[sources/convnext]] の最大の発見は「**性能差の最大の単一要因が訓練レシピだった**」（レシピ変更だけで ResNet-50 が 76.1 → 78.8）。**古いレシピの数字と新しい数字を並べてはいけない**
4. **入力解像度** — 224² / 384² / 512² で 1〜2 ポイント動く
5. **アーキテクチャの定義** — 静的カーネルか、内容依存の適応性を持つか

---

## 関連ページ

- [[sources/convnext-v2]] / [[entities/convnext-v2]] — 基準点となる ConvNeXt V2（88.9%、公開データのみ）
- [[questions/convnext-v1-vs-v2-architecture]] — V1 と V2 の差分（GRN + FCMAE の co-design）
- [[sources/nfnet]] / [[entities/nfnet]] — 本ページで引用した NFNet の原典（ingest 済み）
- [[sources/maxvit]] / [[translations/maxvit]] — NFNet-F4+ 89.20 を含む大規模事前学習の比較表の出典
- [[concepts/convolutional-neural-network]] — CNN の系譜と「CNN は終わったのか」の現在地
- [[sources/convnext]] — 「訓練レシピが最大の交絡変数」という比較の作法
- [[entities/dinov3]] — ConvNeXt を蒸留先として使う、CNN の現在の主戦場
- [[entities/swin-transformer]] / [[entities/maxvit]] — 「相手の性質を輸入する」構図の他の例

## 今後の ingest 候補

本ページの記述を一次情報に格上げするために有用な原典。

- **InternImage**（Wang et al., CVPR 2023） — 本ページ最大の未検証部分。DCNv3 と 89.6% の詳細
- ~~NFNet~~ → **[[sources/nfnet]] として ingest 済み**（2026-06-17）
- **EfficientNet / EfficientNetV2**（Tan & Le） — 比較表に頻出するが未取り込み
- **CoAtNet**（Dai et al., NeurIPS 2021） — [[entities/maxvit]] の最大の比較対象であるハイブリッド
