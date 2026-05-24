# Log — Computer Vision Wiki

時系列の append-only ログです。`## [YYYY-MM-DD] <op> | <title>` のフォーマットで先頭が始まる行を必ず保ってください（`grep "^## \[" log.md | tail -10` で直近を取り出せるようにするため）。

`<op>` は次のいずれか：
- `ingest` — 原典の取り込み
- `query` — 質問への回答（成果物を wiki 化した場合）
- `lint` — 健康診断
- `schema-update` — CLAUDE.md の改訂

---

## [2026-05-24] schema-update | wiki セットアップ初期化

- 作成: `CLAUDE.md`（スキーマ）, [[index]], [[log]], [[overview]]
- 構成: `raw/{papers,articles,images,assets}` / `wiki/{sources,translations,concepts,entities,questions}`
- 追加ルール:
  - 論文・記事（appendix を除く）の ingest 時は [[translations]] に全文翻訳（一文ずつ正確に、要約しない）を作成
  - 原典の画像リンクは [[translations]] / [[sources]] の両方で必要に応じて引用
  - [[translations]] 以外のすべてのページで、略称展開・難概念補足・初学者向けの平易な解説を必須化
- メモ: 最初の原典 ingest を待つ状態。

## [2026-05-24] ingest | Emerging Properties in Self-Supervised Vision Transformers (DINO)

- 取り込み: `raw/papers/Emerging Properties in Self-Supervised Vision Transformers.md`
- ダウンロード: `raw/assets/dino/` に Fig 1-7 を保存（attn6.png, x1.png, 1935.png, fig4-sup-857mask-head1.png, x2.png, x3.png, x4.png）
- 作成:
  - [[translations/dino-emerging-properties-in-self-supervised-vit]] — Abstract + §1-6 の全文和訳（Appendix A-H, References, Acknowledgement は除外）
  - [[sources/dino-emerging-properties-in-self-supervised-vit]] — 初学者向け要約ページ（一言まとめ／背景／提案手法／実験結果／意義／限界／用語）
  - [[concepts/self-supervised-learning]] — SSL 全般。momentum encoder と multi-crop もここに統合
  - [[concepts/vision-transformer]] — ViT アーキテクチャ・命名規約・CNN との比較
  - [[concepts/knowledge-distillation]] — KD の基礎と DINO 流 self-distillation との対比
  - [[concepts/knn-evaluation-protocol]] — SSL 評価における k-NN プロトコル
  - [[entities/dino]] — DINO 手法スペックシート
  - [[entities/imagenet]] — ImageNet データセット
- 更新: [[index]]（全新規ページ追記、略称表に ViT/SSL/KD/DINO/EMA/k-NN/ILSVRC 等を登録）、[[overview]]（SSL 系譜と Transformer アーキテクチャ系譜に DINO を追記）
- メモ:
  - 「機械的なまとめ」を避けるため、source/concepts/entities では全略称を展開して短い意味付けを併記、誘導補足（"なぜ崩壊回避が必要か" など）を多用した
  - momentum encoder と multi-crop は当面 [[concepts/self-supervised-learning]] に集約。別の SSL 手法を ingest して粒度感が変わったら独立ページに切り出す候補
  - 後続の自然な拡張候補: BYOL / MoCo / SwAV / SimCLR / DINOv2 / MAE の ingest または専用ページ化

## [2026-05-24] schema-update | 画像キャプション書式を `<figure>` + `<figcaption>` に統一

- CLAUDE.md §7 に §7.1「キャプション書式（Obsidian 対応）」を追加
- 理由: Obsidian は Markdown 標準の alt テキスト `![caption](path)` をキャプションとして表示せず、別途 CSS スニペットが必要。ユーザー側の追加設定なしで Reading view にキャプションを出すため、HTML の `<figure>` + `<figcaption>` で Markdown 画像記法を包む方式を採用。
- 既存ページの追従:
  - [[translations/dino-emerging-properties-in-self-supervised-vit]] の図 1〜7 を新書式に変換
  - [[sources/dino-emerging-properties-in-self-supervised-vit]] の図 2 と図 1（再掲）を新書式に変換
- 補足: `<figcaption>` 内で `$...$` インライン LaTeX が確実に動かないため、Unicode 添字（x₁ 等）や簡易表記（224^2 等）に置き換えるルールも明記

## [2026-05-24] ingest | DINOv2: Learning Robust Visual Features without Supervision

- 取り込み: `raw/papers/DINOv2_ Learning Robust Visual Features without Supervision.md`
- ダウンロード: `raw/assets/dinov2/` に Fig 1-10 を保存（new-figure-1/7/8/9/10.jpg, x1/x2/x3/x10/x11.png）
- 作成:
  - [[translations/dinov2-learning-robust-visual-features-without-supervision]] — Abstract + §1-10 の全文和訳（Appendix A-C, References, Acknowledgments は除外）
  - [[sources/dinov2-learning-robust-visual-features-without-supervision]] — 初学者向け要約ページ
  - [[concepts/masked-image-modeling]] — MIM 全般（BEiT → MAE → SimMIM → iBOT の系譜）
  - [[concepts/foundation-model]] — 基盤モデルの定義・歴史・批判
  - [[concepts/weakly-supervised-pretraining]] — 弱教師あり事前学習（CLIP 系）と SSL との対比
  - [[entities/ibot]] — iBOT 手法（DINO + MIM）
  - [[entities/dinov2]] — DINOv2 モデル群スペックシート
  - [[entities/lvd-142m]] — 142M 画像のキュレーション済みデータセット
  - [[entities/clip]] — CLIP / OpenCLIP（弱教師あり代表、DINOv2 主要ベースライン）
- 更新: [[index]]（全新規ページ追記、略称表に DINOv2/iBOT/MIM/MAE/BEiT/CLIP/OpenCLIP/LVD-142M/SK/KoLeo/SwiGLU/FFN/FSDP/DDP/DPT/PUE/WSL 等を登録）, [[log]], [[overview]]（Transformer 系譜・SSL 系譜・データセットに DINOv2/iBOT/LVD-142M を追加、weakly-supervised セクション追加）, [[concepts/self-supervised-learning]]（MIM 系統と DINOv2 への接続強化）, [[entities/dino]]（後継への参照追加）, [[entities/imagenet]]（ImageNet-22k の存在感を補強）
- メモ:
  - 略称展開・初学者向け補足を [[sources]] と [[concepts]]、[[entities]] で徹底
  - 9 新規 + 6 更新の大規模 ingest。1 件の論文取り込みでこの規模になるのは「DINOv2 が DINO/iBOT/CLIP/MIM/foundation model 等の複数の系譜の合流点だから」という性質による
  - 後続候補: BYOL / MoCo / SwAV / MAE 単独論文 / SAM / DINOv3 等

## [2026-05-24] ingest | DINOv3

- 取り込み: `raw/papers/DINOv3.md`
- ダウンロード: `raw/assets/dinov3/` に Fig 1〜19 を 18 枚保存（簡略名 fig1.png〜fig19-satellite-comp.jpg に統一）
- 作成:
  - [[translations/dinov3]] — Abstract + §1-10 の全文和訳（References, Appendix A 以降は除外）
  - [[sources/dinov3]] — 初学者向け要約ページ。DINOv2 との差分、Gram anchoring の直観、衛星版を強調
  - [[concepts/gram-anchoring]] — Gram 行列を過去の teacher に合わせる正則化。dense feature 劣化問題への解決法
  - [[concepts/rotary-position-embeddings]] — RoPE。NLP 標準を vision に持ち込み、可変解像度対応に必須
  - [[entities/dinov3]] — DINOv3 モデル群スペックシート（ViT-S〜7B + ConvNeXt + dino.txt）、register tokens / multi-student distillation も内包
  - [[entities/lvd-1689m]] — Instagram 17B からの階層 k-means キュレーション、LVD-142M の 12 倍規模
  - [[entities/sat-493m]] — Maxar 衛星画像 493M、DINOv3 衛星版用
  - [[entities/perception-encoder]] — Meta PE / PEcore / PEspatial（DINOv3 主要競合・弱教師あり）
  - [[entities/siglip]] — Google SigLIP / SigLIP 2（DINOv3 主要競合・弱教師あり）
- 更新: [[index]]（全新規ページ追記、略称表に Gram Anchoring/RoPE/PE/SigLIP/LVD-1689M/SAT-493M/CNX/AM-RADIO/EVA-CLIP/V-JEPA/JEPA/VGGT/DAv2/DPT/SAM/LiT/dino.txt/TTA/OOD/WRI/AIMv2/Franca/Web-DINO/FlashAttention/register tokens 等を登録）, [[log]], [[overview]]（系譜・データセットに DINOv3 を反映、Gram anchoring の項目追加）, [[concepts/self-supervised-learning]]（ハイブリッド型に DINOv3 を追加）, [[entities/dinov2]]（後継 DINOv3 への参照追加）, [[entities/lvd-142m]]（後継 LVD-1689M への参照追加）
- メモ:
  - 9 新規 + 6 更新の大規模 ingest（DINOv2 と同等規模）。複数の競合モデル群（PE/SigLIP）と新規概念（Gram anchoring/RoPE）が一気に登場するため
  - register tokens は独立 concept にせず entities/dinov3 内に集約。Darcet et al. の register 論文を ingest したら昇格を検討
  - multi-student distillation も独立 concept にせず entities/dinov3 内に集約
  - AM-RADIO / Franca / Web-DINO / AIMv2 / EVA-CLIP / V-JEPA 2 / VGGT / DAv2 / Plain-DETR / Mask2Former / ViT-Adapter は独立 entity にせず、entity 内の短い言及にとどめた。再登場頻度を見て昇格を検討
  - 衛星画像セクション（§8）は wiki に独立ドメインページを作るか検討の余地。今後地理空間関連の ingest が増えたら考える
