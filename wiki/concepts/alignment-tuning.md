---
type: concept
aliases: [alignment tuning, アラインメント・チューニング, アラインメントチューニング]
tags: [training-strategy, fine-tuning, contrastive-learning, mllm, dense-prediction]
related: ["[[contrastive-learning]]", "[[foundation-model]]", "[[knowledge-distillation]]", "[[masked-image-modeling]]"]
sources: ["[[sources/perception-encoder]]"]
updated: 2026-05-28
---

# Alignment Tuning（アラインメント・チューニング）

## 何か

**Alignment tuning** は、[[sources/perception-encoder]]（Bolya et al., NeurIPS 2025）が導入したファインチューニング戦略。**「事前学習済みエンコーダの中間層に眠っている一般特徴量を、ネットワークの最終層（出力）まで引き出す（lift）」** ことを目的とする短いファインチューニング段階を指す。

要点を 1 文で言うと: **「モデルにすでに存在する強い特徴を引き出すための仕上げ作業」** であり、**新しい知識を蒸留することが目的ではない**。

## なぜ必要か

CLIP スタイルの大規模対比学習モデル（[[concepts/contrastive-learning]]）には、興味深い病理がある：

- 十分にスケールした対比モデルは、**中間層に** OCR、VQA、grounding、検出、深度推定、追跡など、本来そのモデルが訓練していない多様な下流タスク用の特徴量を **自然に学んでしまう**。
- しかし、**最終層**（CLIP loss が直接適用される層）では、これらの一般特徴は「圧縮されすぎて」消えてしまう。
- 結果として、**最終層の出力をそのまま使う通常のファインチューニングでは、中間層の宝物を活かせない**。

> **補足: なぜ中間層に育つのか** — CLIP 損失は画像-テキスト対のグローバル類似度しか要求しないため、最終層は「グローバル意味」に圧縮される。一方、ネットワークの内部では空間情報・局所構造・OCR 的詳細など、最終出力には不要だが下流タスクには有用な情報を「持ったまま」処理が進む。スケールが小さいうちはこの中間表現も貧弱だが、十分大きくなると驚くほど豊かになる。

この **アラインメント問題（alignment problem）** に対する解が alignment tuning。

## どう動くか

### 共通原則

1. **凍結した「教師」を用意する**。教師は外部モデル（例: SAM）でも、自分自身の凍結スナップショット（例: PEcore G の凍結層 41）でもよい。
2. **短いファインチューニング段階を走らせる**（事前学習比で 1〜10% のサンプル数）。
3. 教師に **最終層出力をマッチさせる** 損失を加える。
4. 重い正則化（DropPath, LayerScale, マスキング）で「忘却」を防ぐ。

### PE での 2 つの具体例

#### 1. 言語アラインメント（PElang）

- **教師の役**: 事前学習済み Llama3.2 3B decoder（テキストのみ）
- **接続**: 2 層 MLP で視覚エンコーダと decoder を接続
- **学習信号**: 次トークン予測損失（OCR Q&A、キャプション化、視覚 Q&A、動画 Q&A の混合データ、70M サンプル）
- **両方 unfrozen** で訓練し、最後に視覚エンコーダのみ抽出
- **正則化**: PEcore の最後の 3 層を破棄、LayerScale + DropPath

**注目すべき結果**: 訓練データに grounding（領域特定）が含まれなくても、RefCOCO grounding 性能が大幅向上。これは PEcore の中間層に既に grounding 能力があり、alignment tuning がそれを末端に「持ち上げた」証拠。

#### 2. 空間アラインメント（PEspatial）

dense タスクは「局所性（locality）」と「高レベル意味」の両方が必要。PE では 2 教師を組み合わせる：

| 教師 | 目的 | 与える信号 |
|---|---|---|
| **PEcore G 自身の凍結層 41 特徴量** | 高レベル空間特徴の維持 | 自己蒸留（cosine 類似度 etc.） |
| **SAM 2.1 mask logits** | 局所性の回復 | 32×32 グリッドで連結したマスク確率マップ |

**重い正則化**: DropPath、LayerScale、75% パッチマスキング。

**自己蒸留と外部教師の併用** が肝。SAM 単独だと意味論を失い、自己蒸留単独だと局所性が不足。

## 既存手法との違い

### 知識蒸留（[[concepts/knowledge-distillation]]）との対比

| 軸 | 知識蒸留 | Alignment Tuning |
|---|---|---|
| 目的 | **教師の知識を student に注入** | **student に既にある特徴を引き出す** |
| 教師の役割 | 答えを教える | アンカーを与える |
| データ規模 | 大きい（教師の知識ほぼ全部を移植） | 小さい（事前学習の 1〜10%） |
| 教師の必要性 | 強い教師モデルが必須 | **自分自身の凍結コピーで十分** |

両者は技術的には似た損失を使うが、**世界観が逆**。蒸留は「持っていないものを受け取る」、alignment は「持っているものを表に出す」。

### LoRA / PEFT との対比

[[concepts/parameter-efficient-fine-tuning]] は「**重みの大部分を凍結したまま小さな修正を加える**」のに対し、alignment tuning は **全パラメータを動かす**（PE では encoder と decoder の両方を unfrozen で訓練）。データ量も alignment tuning は PEFT より遥かに大きい（70M サンプル）。

### Linear probing との対比

Linear probing は「**特徴量を変えずに線形分類器を学習**」する評価手法。Alignment tuning は逆に「**最終層に出てくる特徴量自体を変える**」。

### ファインチューニング一般との対比

通常のファインチューニングは「**下流タスクの正解にモデルを合わせる**」。Alignment tuning は「**自分の中間層に最終層を合わせる**（または外部の dense-aware 教師に合わせる）」。下流タスク特有の正解ラベルを必要とせず、自己整合性を高めるだけで dense 性能が上がる点が新しい。

## 既存研究のヒント

PE は alignment tuning を最初に **系統的かつ実用的に定式化** したが、関連する着想は分散していた：

- **画像着色** [Zhang et al., 2016] — 中間層の特徴が下流タスクに有用と早期に指摘
- **次トークン予測モデル** [Chen et al., 2020] — autoregressive vision で中間層が最良との発見
- **CLIP の限定的観察** [Sun et al., 2024] — CLIP モデルでも中間層が良いことを示唆
- **AM-RADIO** [Ranzinger et al., 2024] — 複数 foundation model を蒸留統合する agglomerative model
- **DINOv3 の Gram anchoring**（[[concepts/gram-anchoring]]） — 過去の自分の Gram 行列でアラインする手法
- **[[entities/internvl-1-5\|InternVL 1.5]] の InternViT V1.2 → V1.5（2024）** — 「最後 3 層を削除」（48 → 45 層）して MLLM タスクで性能向上。「最終層が CLIP 損失目的に強く整列し、ローカル情報よりグローバル整列を優先する」と分析。**PE の中間層特徴発見の約 1 年早い先取り**
- **[[entities/internvl-2-5\|InternVL 2.5]] のセグ/分類アブレーション（2024-12）** — InternViT のバージョン更新で **linear probing が継続的に低下、attention pooling / head tuning は維持または改善**。「最終層の線形分離性が低下しながら、複雑・非線形な意味情報を捕捉」と明確に観察、Δ（linear と attn の差）が 3.0 → 6.7 と拡大。**PE と独立に同じ現象を発見**

PE の貢献は、これらの分散した知見を **複数モデルクラス（contrastive / SSL / captioning）で同時実証** し、**alignment tuning という統一フレームに整理** したこと。同時期の InternViT 系統も独立に同じ現象を観察しており、**「対比学習 + NTP 共同訓練の VFM では中間層特徴が線形性を失い、リッチな意味を獲得する」** という現象は **2024 年に複数チームが独立に確認した普遍的事実** と言える。

## 限界・注意点

- **「すでに特徴がある」前提**: alignment tuning は中間層に豊かな特徴がない小型モデルでは効果が薄い。**大規模事前学習が前提条件**。
- **タスクごとに教師が違う**: 言語タスクは LLM decoder、空間タスクは SAM + 自己蒸留と、用途別の教師設計が必要。「全タスク統一の alignment tuning」はまだない。
- **理論的説明の不在**: なぜ対比学習で中間層に多目的特徴が育つかの理論的解釈はなく、経験的観察のみ（PE 論文 Appendix C.5 でいくらか解析）。
- **モデル運用の複雑化**: PE の場合、PEcore / PElang / PEspatial の 3 バリアントを使い分ける必要があり、単一エンコーダの単純さは失われる。

## 既存 wiki との接続

- [[sources/perception-encoder]] — alignment tuning を初めて系統的に提示した論文
- [[entities/perception-encoder]] — PE モデルファミリーの詳細
- [[concepts/foundation-model]] — alignment tuning は基盤モデル時代の新ファインチューニング戦略
- [[concepts/contrastive-learning]] — PEcore の事前学習目的、alignment tuning が対象とする「最終層が一般特徴を出力しない」病理の発生源
- [[concepts/knowledge-distillation]] — alignment tuning が技術的に借用する蒸留損失の親概念
- [[concepts/gram-anchoring]] — DINOv3 が使う、目的は近いが手法が異なる正則化
- [[concepts/parameter-efficient-fine-tuning]] — 「少量更新」という意味で alignment tuning と対極にある軽量化戦略

## 今後の展望

- **他の foundation model への適用**: SigLIP 2 / DINOv3 などにも適用可能か？
- **言語と空間以外の alignment**: 動画時間軸、3D 構造などへの拡張は？
- **小型化**: PE は G/14（2B）でデモしたが、小型モデルでも有効か？（PEcore B/L での実験は Appendix にあり、効果は限定的）
- **理論基盤**: 「対比学習が中間層に育てる一般特徴」のメカニズム解明は、今後の foundation model 研究の中心トピックになりうる
