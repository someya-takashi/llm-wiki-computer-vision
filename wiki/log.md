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

## [2026-05-25] ingest | MAE: Masked Autoencoders Are Scalable Vision Learners

- 取り込み: `raw/papers/Masked Autoencoders Are Scalable Vision Learners.md`
- ダウンロード: `raw/assets/mae/` に Fig 1〜11 を 10 枚保存（x1〜x12.png）
- **特例**: ユーザー指示により **Appendix A〜C も翻訳対象に含めた**（References のみ除外）。CLAUDE.md §4 のデフォルトと異なる扱い
- 作成:
  - [[translations/mae]] — Abstract + §1-6 + Appendix A-C の全文和訳
  - [[sources/mae]] — 初学者向け要約。75% マスクの直観・非対称設計・linear vs fine-tune の解離を強調
  - [[concepts/denoising-autoencoder]] — DAE の理論的枠組み。MAE/BERT MLM/拡散モデルすべての祖先として位置づけ
  - [[entities/mae]] — MAE モデル群スペックシート（ViT-B/L/H、公開モデル情報）
- 更新: [[index]]（新規ページ追記、略称表に MAE/DAE/MLM 登録）, [[log]], [[overview]]（MIM 系譜に MAE を明示）, [[concepts/masked-image-modeling]]（MAE 節を大幅深化）, [[concepts/self-supervised-learning]]（MAE 関連を補強）, [[entities/dinov2]]（MAE との対比を補強）
- メモ:
  - 4 新規 + 6 更新の中規模 ingest
  - DINO 3 部作で常に参照されていた MAE をようやく深く扱えた。後続の iBOT/DINOv2/DINOv3 のページからの逆参照が機能するようになり、wiki の縦串と横串が初めて完全接続
  - 「DAE は現代 SSL のほぼすべての祖先」という視点を [[concepts/denoising-autoencoder]] で確立。今後拡散モデル系を ingest する際にも基礎として機能するはず
  - 後続候補: iBOT 単独論文、CLIP 単独論文、SAM、BYOL/SimCLR/MoCo v3 など対比学習系

## [2026-05-25] ingest | iBOT: Image BERT Pre-Training with Online Tokenizer

- 取り込み: `raw/papers/iBOT _ Image BERT Pre-Training with Online Tokenizer.md`
- ダウンロード: `raw/assets/ibot/` に Fig 1〜19 を 18 枚保存（x2〜x24.png）
- **特例**: ユーザー指示により **Appendix A〜G も翻訳対象に含めた**（References のみ除外）。MAE ingest と同様の扱い
- 作成:
  - [[translations/ibot]] — Abstract + §1-6 + Appendix A-G の全文和訳（Appendix A 擬似コード、B マルチクロップ統合、C 実装詳細、D 追加結果、E 追加アブレーション、F 代替トークナイザ、G 可視化を含む）
  - [[sources/ibot]] — 初学者向け要約。online tokenizer の核心、DINO/MAE との関係、部位レベル意味性の創発を強調
  - [[concepts/online-tokenizer]] — iBOT の核心アイデア「teacher 自身を MIM の視覚トークナイザとして動的に使う」の独立解説。DINOv2/v3 の損失設計の中核
- 更新:
  - [[entities/ibot]] — DINOv2 経由の二次情報で書かれていたページを、原典 ingest に基づいて**大幅 rewrite**（一次情報化）。詳細な設計選択、マルチクロップ統合の苦労、創発する部位レベル意味性、限界等を補強
  - [[index]] — sources/translations/concepts/entities すべてに iBOT 追加、略称表に online tokenizer/dVAE 登録
  - [[log]]、[[concepts/masked-image-modeling]]（iBOT 節の深化）、[[entities/dinov2]]（sources/ibot への参照追加）、[[entities/dino]]（iBOT を直接の後継として明示）
- メモ:
  - 3 新規 + 6 更新の中規模 ingest
  - これまで「DINOv2 ページ経由の二次情報」だった iBOT を、原典ベースの一次情報に格上げした。これで **DINO → iBOT → DINOv2 → DINOv3 の系譜がすべて原典 ingest 済み**となり、SSL DINO 系統の wiki が完全に閉じた
  - 「online tokenizer」は独立 concept に昇格。BEiT/DINOv2/DINOv3 すべてが言及する重要概念で、将来 BEiT を ingest する際にも基盤として機能するはず
  - 後続候補: CLIP / BYOL / SimCLR / MoCo v3 / SAM / BEiT 等の対比学習・SSL 関連、または Swin Transformer / ConvNeXt 等のアーキテクチャ系

## [2026-05-25] ingest | CLIP: Learning Transferable Visual Models From Natural Language Supervision

- 取り込み: `raw/papers/Learning Transferable Visual Models From Natural Language Supervision.md`
- ダウンロード: `raw/assets/clip/` に Fig 1〜23 + SST2 例画像を 22 枚保存
- **特例**: ユーザー指示により **Appendix A〜F も翻訳対象に含めた**（References のみ除外）。MAE / iBOT と同様の扱い
- 作成:
  - [[translations/clip]] — Abstract + §1-9 + Appendix A-F の全文和訳（線形プローブ評価、ゼロショット予測、重複検出器、YFCC100M アブレーション、選択タスク結果、モデルハイパーパラメータを含む）
  - [[sources/clip]] — 初学者向け要約。WIT の規模、対比学習、ゼロショット転移、頑健性、§7 のバイアス問題までを丁寧に解説
  - [[concepts/contrastive-learning]] — 対比学習の独立 concept 化。InfoNCE 損失、SimCLR/MoCo/CLIP/DINO 系統の基盤として
  - [[concepts/zero-shot-transfer]] — CLIP が CV に持ち込んだゼロショット転移の独立 concept 化。プロンプトエンジニアリング、effective robustness 含む
  - [[entities/wit-400m]] — CLIP の訓練データ WIT（4 億画像-テキスト対、OpenAI 内部）
- 更新:
  - [[entities/clip]] — DINOv2/v3 ingest 時に書かれた二次情報を、CLIP 原典 ingest に基づいて**大幅 rewrite**（モデル詳細、OpenCLIP との対比、産業応用への影響、限界の明示的言及）
  - [[index]] — sources/translations/concepts/entities すべてに CLIP 追加、略称表に InfoNCE/WIT/BPE/zero-shot/prompt engineering を登録
  - [[log]]
  - 残り: [[overview]], [[concepts/weakly-supervised-pretraining]], [[concepts/foundation-model]]（次の編集で対応）
- メモ:
  - 5 新規 + 6 更新の大規模 ingest（DINOv2/v3 と同等規模）
  - 「contrastive learning」「zero-shot transfer」は wiki 全体で頻繁に言及されながら、これまで独立 concept ページがなかった重要 gap。CLIP ingest を機に解消
  - WIT は OpenAI 内部データセットで非公開だが、後続の LAION/WebLI 等の発端なので独立エンティティ化
  - CLIP entity は 3 度目の major update（DINOv2 → DINOv3 → CLIP 自身の原典）。これで一次情報化が完了
  - 後続候補: SAM (Segment Anything) / Stable Diffusion / LLaVA / BLIP-2 / BYOL / MoCo v3 / BEiT / SwAV / DETR / Mask2Former 等

## [2026-05-26] ingest | SAM: Segment Anything

- 取り込み: `raw/papers/Segment Anything.md`
- ダウンロード: `raw/assets/sam/` に Fig 1〜17（SA-1B 例画像含む）を 17 枚保存
- **特例**: ユーザー指示により **Appendix A〜G も翻訳対象に含めた**（References のみ除外）。MAE / iBOT / CLIP と同様の扱い
- 作成:
  - [[translations/segment-anything]] — Abstract + §1-8 + Appendix A-G の全文和訳（SAM/タスクの詳細、自動マスク生成、RAI、実験実装、人手評価設計、データセット・モデルカード、アノテーションガイドラインを含む）
  - [[sources/segment-anything]] — 初学者向け要約。promptable segmentation の意義、データエンジンの 3 段階、SA-1B のスケール感、ゼロショット転移結果、§8 の限界、3 大 CV foundation model 系統での位置づけを強調
  - [[concepts/promptable-segmentation]] — SAM が定義した新タスクパラダイムの独立 concept。プロンプト種類、曖昧性対応、interactive seg との違い、後続研究まで
  - [[entities/sam]] — SAM モデル群スペックシート（ViT-B/L/H、Apache 2.0、訓練レシピ、推論コード例、後継モデル一覧）
  - [[entities/sa-1b]] — SA-1B データセット（11M 画像 × 1.1B マスク、データエンジン構築、地理分布、公開条件、影響）
- 更新:
  - [[index]] — sources/translations/concepts/entities すべてに SAM/SA-1B 追加、略称表に SAM/SA-1B/RITM/ViTDet/LVIS/COCO/NMS/IoU/mIoU/focal loss/dice loss/MIAP/RAI/stuff/things/modal/amodal/BSDS500/ODS/OIS/R50/promptable segmentation 登録
  - [[log]]
  - 残り: [[overview]], [[concepts/foundation-model]], [[entities/clip]]（CLIP テキストエンコーダ使用言及）, [[entities/mae]]（SAM 画像エンコーダ初期化への言及）の更新を次の編集で実施
- メモ:
  - 5 新規 + 6 更新の大規模 ingest（DINOv2/v3/CLIP と同等規模）
  - SAM は **CV foundation model 第 3 の系統**として位置づけ（CLIP=WSL, DINO/MAE/iBOT=SSL, SAM=モデル支援アノテーション + 教師あり）。Discussion §8 で著者自身が CRFM の "foundation model = SSL" 前提を明示的に修正している点を強調
  - "promptable segmentation" は SAM 独自の重要概念で、prompt engineering を CV のセグメンテーションタスクに持ち込んだ初の枠組みのため独立 concept に昇格
  - SA-1B は SAM の中核資産であり、公開された世界最大のセグメンテーションデータセット（Open Images の 400 倍マスク）なので独立 entity 化
  - SAM 2（動画拡張版）、MobileSAM、HQ-SAM、Grounded-SAM 等の派生は entities/sam 内の言及にとどめた。SAM 2 は raw/papers/ にも置かれているので、別途 ingest 候補
  - 後続候補: SAM 2（動画版）/ Stable Diffusion / LLaVA / BLIP-2 / ViT 原典 / SigLIP / Perception Encoder / BEiT 等

## [2026-05-26] ingest | SAM 2: Segment Anything in Images and Videos

- 取り込み: `raw/papers/SAM 2_ Segment Anything in Images and Videos.md`
- ダウンロード: `raw/assets/sam-2/` に Fig 1〜17 を 16 枚保存（PVS タスク図、SAM 2 アーキテクチャ、SA-V サンプル、データエンジン、ゼロショット結果等）
- **特例**: ユーザー指示により **Appendix A〜G も翻訳対象に含めた**（References のみ除外）。MAE / iBOT / CLIP / SAM v1 と同様の扱い
- 作成:
  - [[translations/sam-2]] — Abstract + §1-9 + Appendix A-G の全文和訳（PVS タスク詳細、限界、SAM 2 アーキテクチャ詳細、訓練レシピ、データエンジン詳細、ゼロショット実験詳細、VOS 比較、モデル・データ・アノテーションカードを含む）
  - [[sources/sam-2]] — 初学者向け要約。PVS タスク、streaming memory 3 構造（空間 + プロンプト + object pointer）、データエンジン 3 phase、Hiera 採用の意味、SA-V スケール感、SAM v1 との詳細比較を強調
  - [[entities/sam-2]] — SAM 2 モデル群スペックシート（Hiera-T/S/B+/L 4 種、訓練レシピ、推論コード例、SAM v1 との対比表、派生モデル一覧）
  - [[entities/sa-v]] — SA-V データセット（50.9K 動画 × 642.6K masklet × 35.5M マスク、CC by 4.0、データエンジン詳細、地理分布、val/test を Phase 1 でアノテートした設計判断）
  - [[entities/hiera]] — Hiera 階層型 ViT（独立 entity 新規作成）。Pool attention、MAE 互換性、SAM 2 採用の意味、Swin/MViT との比較
- 更新:
  - [[index]] — sources/translations/entities すべてに SAM 2/SA-V/Hiera 追加、略称表に SAM 2/SA-V/Hiera/PVS/masklet/VOS/iVOS/𝒥&ℱ/2D-RoPE/FlashAttention/MOSE/LVOS/DAVIS/YouTube-VOS/XMem/Cutie/FPN/GRU 登録
  - [[log]]
  - 残り: [[overview]], [[entities/sam]], [[entities/sa-1b]], [[concepts/promptable-segmentation]], [[concepts/rotary-position-embeddings]], [[concepts/foundation-model]], [[entities/mae]] の更新を次の編集で実施
- メモ:
  - 5 新規 + 7 更新の大規模 ingest（SAM v1 と同等規模）
  - SAM 2 は **動画 foundation model の de facto 標準**として位置づけ。SAM v1 の上位互換（画像でも 6× 高速 + 高精度）でもあるため、SAM v1 の役割は徐々に SAM 2 に置き換わる方向
  - "Hiera" を独立 entity に昇格。SAM 2 で初登場だが、今後の動画系・密予測系 foundation model（特に Meta 系）で広く使われると予想
  - "PVS" は [[concepts/promptable-segmentation]] の動画拡張版として既存 concept に統合（独立 concept にはしない）。これは「画像 = 1 フレーム動画」という SAM 2 の統一設計を反映
  - "masklet" は SA-V/SAM 2 固有用語として entities ページ内で定義、独立 concept にはしない
  - XMem++ / Cutie / DEVA など強力な VOS ベースラインは entities/sam-2 内で簡略言及。再登場頻度を見て独立化を検討
  - 後続候補: ViT 原典 / SigLIP / Perception Encoder / V-JEPA / V-JEPA 2 / BEiT / Stable Diffusion / LLaVA など

## [2026-05-26] ingest | SAM 3: Segment Anything with Concepts

- 取り込み: `raw/papers/SAM 3_ Segment Anything with Concepts.md`
- ダウンロード: `raw/assets/sam-3/` に Fig 1〜9 を 8 枚保存（PCS タスク図、SAM 3 アーキテクチャ、データエンジン、SA-Co 例、対話的評価、ドメイン適応）
- **特例**: 原典ファイルの取り込み制約により、Appendix の実体（A.2 以降、B-H）が含まれていなかった（セクションラベルのプレースホルダのみ）。ユーザー判断で **本文（§1-9）+ Appendix A.1 Presence Token 部分のみ** を ingest（CLAUDE.md §4 デフォルトに近い対応）
- 作成:
  - [[translations/sam-3]] — Abstract + §1-9 + Appendix A.1 冒頭部分の和訳。frontmatter で取り込み制約を明記
  - [[sources/sam-3]] — 初学者向け要約。PCS タスク、presence head の核心、データエンジン 4 phase（AI verifier）、SA-Co、SAM 3 Agent パターン、SAM 1/2/3 進化表を強調
  - [[concepts/promptable-concept-segmentation]] — PCS タスクの独立 concept（新規）。PVS との対比、プロンプト型、評価メトリクス、後続研究の予想まで
  - [[entities/sam-3]] — SAM 3 モデル群スペックシート。アーキテクチャ詳細（共有 PE、DETR detector、presence token、tracker、matching）、訓練ステージ、SAM 1/2/3 進化表、限界
  - [[entities/sa-co]] — SA-Co データセット + benchmark の独立 entity。4 つの訓練データ群 + 4 分割の benchmark、データエンジン 4 phase、AI verifier、SA-Co ontology
- 更新:
  - [[index]] — sources/translations/concepts/entities すべてに SAM 3/SA-Co 追加、略称表に SAM 3/SA-Co/PCS/NP/hard negative/exemplar/presence token/AI verifier/MV/EV/cgF₁/pmF₁/IL_MCC/pHOTA/HOTA/DETR/MDETR/MaskFormer/SAM 3 Agent 登録
  - [[log]]
  - 残り: [[overview]], [[entities/sam]], [[entities/sam-2]], [[entities/perception-encoder]], [[concepts/promptable-segmentation]], [[concepts/foundation-model]] の更新を次の編集で実施
- メモ:
  - 5 新規 + 6 更新の大規模 ingest（SAM v1, SAM 2 と同等規模）
  - SAM 3 は **CV foundation model の第 4 軸（コンセプト指定）** として位置づけ。SAM 1/2 の PVS が「インスタンス指定」だったのと並行で、SAM 3 の PCS が「コンセプト指定」を導入
  - 重要な構成判断：**PCS は PVS と並ぶ独立 concept として昇格**（promptable-concept-segmentation.md）。PVS の単なる拡張ではなく、入力（点/ボックス vs 名詞句/exemplar）と出力（1 オブジェクト vs 全インスタンス）が根本的に異なるため
  - **Meta Superintelligence Labs** という新組織が SAM 3 を発表（SAM 1/2 は FAIR）。組織再編後の最初の主要 vision foundation model
  - Perception Encoder（[[entities/perception-encoder]]）が backbone として本格採用された初の主要 foundation model
  - SAM 3 Agent パターン（MLLM が SAM 3 をツールとして使う）は **長期指示表現/推論を SAM 3 のスコープ外に置き、MLLM で補完する** という設計判断の体現。CLIP/SAM/LLM の組み合わせ pattern の集大成
  - データエンジンの AI verifier（Llama 3.2 ベース）は人間アノテーションコスト半減 + 合成データ SYN が HQ に追いつくスケーリングを実証。これは foundation model 時代のデータ収集の根本的進化
  - Appendix の大部分が原典ファイルに含まれなかったため、訓練詳細・hyperparameter・データ統計・実装詳細が未取得。将来 ingest 候補
  - 後続候補: Perception Encoder（PE）の単独原典 ingest、V-JEPA 2、Stable Diffusion、LLaVA、SAM 3 後継モデル等

## [2026-05-26] ingest | SigLIP: Sigmoid Loss for Language Image Pre-Training

- 取り込み: `raw/papers/Sigmoid Loss for Language Image Pre-Training.pdf`（PDF 形式、本文 10 ページ + References）
- **特例 1**: ユーザー方針により **画像はダウンロードしない**（PDF 本体に図が含まれるため）
- **特例 2**: 独立 Appendix が存在しない（本文中に小規模統合）、本文 §1-5 + Acknowledgements が翻訳対象
- 作成:
  - [[translations/siglip]] — Abstract + §1-5 の全文和訳。アルゴリズム 1 擬似コード、4 表、5 図（参照のみ）、訳注付き
  - [[sources/siglip]] — 初学者向け要約。sigmoid loss の数式対比、bias term + β₂=0.95 + chunked impl + 32k 飽和発見の 4 大実装テクニックを強調、CLIP/SigLIP/PE/DINOv3 の対比表
- 更新（major rewrite）:
  - [[entities/siglip]] — **大幅 rewrite**。DINOv3/SAM 3 経由の二次情報を、原典 ingest に基づく一次情報化（[[entities/clip]] と同様のパターン）。SigLiT vs SigLIP の区別、bias term 初期化、β₂=0.95、chunked 実装、4 TPU 効率、ノイズ頑健性、応用 VLM（PaliGemma 等）を網羅
- 更新（軽微）:
  - [[index]] — sources/translations 追加、略称表に SigLIP/SigLiT/mSigLIP/sigmoid loss/WebLI/SO-400M/XM3600/LiT/InfoNCE/big_vision/β₂/chunked implementation 登録
  - [[log]]
  - 残り: [[overview]], [[entities/clip]], [[entities/perception-encoder]], [[concepts/contrastive-learning]], [[concepts/weakly-supervised-pretraining]], [[concepts/foundation-model]] の更新を次の編集で実施
- メモ:
  - 2 新規 + 1 major rewrite + 4-5 軽微更新の中規模 ingest
  - **二次情報からの昇格パターン**: CLIP entity（DINOv2/v3 → 原典）に続いて SigLIP entity も二次情報 → 一次情報化された。この pattern は ingest スキーマの重要な型として確立
  - sigmoid loss は **対比学習の重要な代替軸** として [[concepts/contrastive-learning]] にも反映予定
  - **bias term の決定的重要性**（b=-10, t'=log10 で 10 ポイント差）は実装する人に向けた最重要メッセージ、sources/siglip と entities/siglip の両方で強調
  - **「32k で飽和」発見**は対比学習研究の常識を覆す重要結果、研究のアクセス可能性を根本的に変える意味で強調
  - SigLIP 2 は ingest 未済だが、entities/siglip 内に SigLIP 2 セクションを残置（NaFlex、多言語、dense feature 改善）。将来 ingest 候補
  - 後続候補: SigLIP 2 / Perception Encoder（PE）原典 / EVA-CLIP / MetaCLIP / DFN / V-JEPA 2 / Stable Diffusion / LLaVA など

## [2026-05-27] ingest | SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features

- 取り込み: `raw/papers/SigLIP 2_ Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features.md`
- ダウンロード: `raw/assets/siglip-2/` に Fig 1-6 を 6 枚保存（訓練レシピ概観、多言語検索、NaFlex 比較、PaliGemma 比較、文化的多様性、representation bias）
- 作成:
  - [[translations/siglip-2]] — Abstract + §1-5 + **Appendix A（PaliGemma 全結果）+ Appendix B（NaFlex 全結果）+ Appendix C（文化的多様性・公平性全結果）** の全文和訳。表 6-9 含む、訳注付き
  - [[sources/siglip-2]] — 初学者向け要約。段階的訓練レシピ（4 段階）、4 大追加技法（LocCa / SILC+TIPS / ACID / NaFlex）、結果（RefCOCO +20pt、dense +4-5pt、representation bias -28pt）、CLIP 系 4 年分の発展レシピの統合俯瞰、DINOv3 系との位置関係
- 更新（major）:
  - [[entities/siglip]] — SigLIP 2 セクションを **「ingest 未済プレースホルダ」から「原典ベース一次情報」へ大幅拡張**。段階的訓練レシピの図、4 技法の対応表、新規 g/16 (1B) を含むモデルラインナップ、主要結果数値、LocCa/SILC/TIPS/ACID の各説明
- 更新（軽微）:
  - [[index]] — sources/translations 追加、略称表に SigLIP 2/NaFlex/NaViT/FlexiViT/LocCa/SILC/TIPS/ACID/ACED/MAP head/Gemma/PaliGemma/representation bias/GeoDE/Dollar Street/RefCOCO/DPT/OWL-ViT/Cat-Seg/HierText/SciCap/Screen2Words/TextCaps/TPUv5e 登録
  - [[log]]
  - [[overview]] — SigLIP 2 を独立項目に分割（v1 と並列）し、統合レシピの本質を反映
  - [[concepts/foundation-model]] — SigLIP 2 を独立項目に分割、「CLIP 4 年分の独立改善を 1 モデルに凝集」を強調
  - [[concepts/weakly-supervised-pretraining]] — SigLIP 2 を独立項目に分割、WSL の弱点（dense）が SigLIP 2 で狭まりつつあるが完全には埋まっていない、という最新状況を追加
  - [[concepts/contrastive-learning]] — 「SigLIP 2: 対比学習を超えた統合レシピ」セクション新規追加。「対比学習は必要だが十分ではない」という 2025 年の到達点を明示
  - [[concepts/self-supervised-learning]] — 「2025 年の逆流: WSL が SSL 技法を借用」コメント追加。区分の融合を明示
  - [[concepts/masked-image-modeling]] — 「SigLIP 2 における MIM（WSL への流入）」セクション新規追加。MIM は SSL 専売特許ではない、を強調
  - [[concepts/knowledge-distillation]] — 「データを通じた暗黙的蒸留（ACID / ACED）」セクション新規追加、暗黙的蒸留の本質説明
  - [[entities/perception-encoder]] — SigLIP との対比表を SigLIP 2 を含む 4 列対比に拡張、「損失効率化 vs 統合レシピ vs データ規模化」の戦略対比を明示
- メモ:
  - 2 新規 + 1 major update + 8 軽微更新の大規模 ingest（[[sources/sam-3]] と同等規模）
  - **構造的に重要な ingest**: SigLIP 2 は「CLIP 系統 4 年分の独立改善を統合した、現在の WSL の到達点」として位置づけ。これにより WSL/SSL/MIM/KD/contrastive の **5 系統概念ページすべてに横断的影響** が出るのは初めて
  - **二次情報→一次情報の昇格パターン継続**: SigLIP entity 内の SigLIP 2 セクションを「ingest 未済プレースホルダ（5 項目箇条書き）」から「段階的訓練レシピ + 4 技法対応表 + 新モデル + 結果数値 + LocCa/SILC/TIPS/ACID 各説明」へ拡張。CLIP entity / SigLIP entity に続く第 3 例
  - **「対比学習は十分条件ではない」発見の明示**: [[concepts/contrastive-learning]] に新節を立て、SigLIP 2 が対比学習を **必要だが不十分な基礎レイヤ** として扱い、その上に decoder / 自己蒸留 / マスク予測 / データキュレーションを積層するアプローチを取ったことを記録。これは 2024-25 年の重要な認識転換
  - **MIM/SSL/KD 概念の横断的アップデート**: SigLIP 2 のマスク予測（TIPS 由来）、自己蒸留（SILC 由来）、ACID 蒸留（Udandarao 由来）は、それぞれ MIM / SSL / KD 概念ページに「WSL への流入」「逆流」「現代的応用」として影響。SSL と WSL の境界が 2025 年に大きく曖昧化したことを記録
  - **新概念ページは作らなかった**: LocCa/SILC/TIPS/ACID/NaFlex/MAP head/Gemma 等は entities/siglip.md と sources/siglip-2.md 内で扱い、専用ページは作成せず。概念ページの増殖を抑え既存ページの拡張を選択（多くは「SigLIP 2 にとっての構成要素」であり、独立性が低いため）
  - **公開モデルとして g/16 (1B) が初登場**: SigLIP v1 の最大は So400m、SigLIP 2 で 1B 投入。Foundation model のサイズ競争が WSL 側でも進行
  - **多言語＋ de-bias の社会的インパクト**: representation bias 35.5% → 7.3% は単なるベンチマーク数値以上に、CLIP/SigLIP 系統が抱えていた根本的問題への直接対処として重要
  - **NaFlex は OCR/文書系で大幅優位**: 標準正方形入力では捨てていたアスペクト比情報を活用、HierText/SciCap/Screen2Words 等で顕著改善。LayoutLM 系と協調するか競合するかは未明
  - 後続候補: PE（Perception Encoder）原典 / EVA-CLIP / MetaCLIP / DFN / V-JEPA 2 / Stable Diffusion / LLaVA / PaliGemma 2 / LocCa / SILC / TIPS 原典 など

## [2026-05-27] ingest | A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)

- 取り込み: `raw/papers/A Simple Framework for Contrastive Learning of Visual Representations.md`
- ダウンロード: `raw/assets/simclr/` に fig1.png（性能比較棒グラフ）、fig5.png（拡張アブレーション行列）、fig7.png（モデルスケール比較）、fig9.png（バッチサイズ×エポック数比較）を保存
- 作成:
  - [[translations/simclr]] — Abstract + §1-8 + Appendix A-C の全文和訳（References のみ除外）
  - [[sources/simclr]] — 初学者向け要約ページ（4 コンポーネント設計 / 主要アブレーション / 実験結果 / 限界 / 用語）
  - [[entities/simclr]] — SimCLR モデルエンティティ（4 コンポーネント表 / 主要結果 / 後継手法系統図）
- 更新:
  - [[concepts/contrastive-learning]] — frontmatter sources 追加 + §1「インスタンス識別」内に SimCLR 核心設計サブセクション追加（3 つの発見、射影ヘッドのフロー図、拡張構成の重要性）、関連ページに simclr 追加
  - [[concepts/self-supervised-learning]] — frontmatter sources に simclr 追加
  - [[index]] — sources/simclr、translations/simclr、entities/simclr を追加、略称表に SimCLR/NT-Xent/LARS/MoCo/PIRL/Global BN を登録
- メモ:
  - **「何が効くかを解明した論文」**: SimCLR の貢献は新しい手法の提案というより、既存要素の何が効いていたかを体系的アブレーションで解明したこと。非線形射影ヘッドの発見（+10%以上）が最大のインパクト
  - **二次情報→一次情報昇格**: [[concepts/contrastive-learning]] には以前から SimCLR への言及があったが、源論文を ingest することで一次情報に基づいたより詳細なセクションに置換
  - **既存の contrastive-learning.md との整合**: ページ構造はほぼ既存のまま維持し、Section 1「インスタンス識別」の下に SimCLR のサブセクションを追加する形をとった
  - **新概念ページは作らなかった**: NT-Xent、LARS 等は sources/simclr.md と index 略称表で完結。MoCo のエンティティページも現時点では不要と判断
  - 後続候補: MoCo / BYOL / SimSiam（非対比型 SSL の実装解明系）、SimCLR v2（知識蒸留への発展）など

## [2026-05-27] ingest | Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning (BYOL)

- 取り込み: `raw/papers/Bootstrap Your Own Latent A New Approach to Self-Supervised Learning.md`
- ダウンロード: `raw/assets/byol/` に fig1.png（ImageNet 比較バーチャート）、fig2.png（BYOL アーキテクチャ図）、fig3a.png（バッチサイズ比較）、fig8.png（BYOL スケッチ / 全体アーキテクチャ）を保存
- 作成:
  - [[translations/byol]] — Abstract + §1-6 + Appendix A-B の全文和訳（References のみ除外）
  - [[sources/byol]] — 初学者向け要約ページ（2 ネットワーク構造 / predictor の役割 / 崩壊回避の機構 / アブレーション結果 / SimCLR との詳細比較 / 用語）
  - [[entities/byol]] — BYOL モデルエンティティ（アーキテクチャ表 / 主要結果 / 後継手法系統図）
- 更新:
  - [[concepts/self-supervised-learning]] — frontmatter sources に byol 追加、BYOL の説明文を拡充（predictor + EMA が両方必要、74.3% vs 69.3%）、参考セクションに byol を追加
  - [[index]] — sources/byol、translations/byol、entities/byol を追加、略称表に BYOL/predictor を登録
  - [[log]]
- メモ:
  - **「負例は必須か → 否」への答え**: BYOL の最大の貢献は「predictor + EMA ターゲットの組み合わせで崩壊を防げる」という発見。この結果が SimSiam（EMA も除去）、DINO（cross-entropy + centering）へと続く非対比 SSL の流れを確立した
  - **二次情報→一次情報昇格**: [[concepts/self-supervised-learning]] には以前から BYOL の記述があったが（「predictor + momentum + BN」）、原典 ingest により「predictor + EMA target（どちらか一方では崩壊する）」という正確な知見に更新
  - **τ アブレーションの重要性**: τ=0（直接コピー）→崩壊、τ=1（固定ランダム）→18.8% でも崩壊せず（これが BYOL の着想源）、τ=0.99 が最適という実験結果は直観と理論の双方を補強する重要データ
  - **新概念ページは作らなかった**: predictor / bootstrapping は sources/byol.md と index 略称表で完結。SimSiam（BYOL から EMA も除去）は次の ingest 候補
  - 後続候補: SimSiam（stop-gradient のみで崩壊防止）、MoCo / MoCo v2（momentum encoder 元祖）、DINO 系（BYOL の自己蒸留解釈を完成）、SimCLR v2（知識蒸留）など

## [2026-05-27] ingest | MixMatch: A Holistic Approach to Semi-Supervised Learning

- 取り込み: `raw/papers/MixMatch_ A Holistic Approach to Semi-Supervised Learning.md`
- ダウンロード: `raw/assets/mixmatch/` に fig1.png（ラベル推定図）、fig4.png（SVHN+Extra 比較グラフ）を保存（fig2/fig3 は原典 markdown に URL 未埋め込み）
- 作成:
  - [[translations/mixmatch]] — Abstract + §1-5 + Appendix A-C（記号定義 / CIFAR-10 全数値表 / SVHN 全数値表 / SVHN+Extra 全数値表 / 13 層 ConvNet 結果）の全文和訳（References のみ除外）
  - [[sources/mixmatch]] — 初学者向け要約ページ（ラベル推定 + シャープニング + MixUp の 3 ステップ / アブレーション重要性順位 / 差分プライバシー応用 / SimCLR/BYOL との比較 / 用語）
  - [[entities/mixmatch]] — MixMatch エンティティ（アルゴリズム構成図 / ハイパーパラメータ表 / 主要結果表 / アブレーション表 / 後継手法表）
  - [[concepts/semi-supervised-learning]] — 半教師あり学習の新規 concept ページ（自己教師あり学習との区別、一貫性正則化 / エントロピー最小化 / MixUp の 3 系統、系譜図、SSL 略称衝突問題の明示）
- 更新:
  - [[concepts/self-supervised-learning]] — related に [[semi-supervised-learning]] 追加、参考セクションに「半教師あり学習：SSL 略称の衝突注意」を追加
  - [[index]] — sources/mixmatch、translations/mixmatch、entities/mixmatch、concepts/semi-supervised-learning を追加、略称表に MixMatch/MixUp/Label Guessing/Sharpening/SeSL/Mean Teacher/VAT/FixMatch/Brier スコア/PATE/Wide ResNet/WRN を登録
  - [[log]]
- メモ:
  - **半教師あり vs 自己教師あり**: MixMatch は SSL（Self-Supervised Learning）ではなく SeSL（Semi-Supervised Learning）。wikiで「SSL」という略称を持つ [[concepts/self-supervised-learning]] との混同を避けるため、新規 concept ページ [[concepts/semi-supervised-learning]] を作成して区別を明示した
  - **MixUp が最大の貢献要素**: アブレーション（表 4）でラベルあき・なし横断 MixUp を除去すると 11.08% → 39.11%（+28pt）という最大の劣化。シャープニング（+16pt）、K 回平均（+5pt）より重要
  - **シャープニングは DINO へ**: MixMatch のシャープニング関数（温度 T=0.5）の思想は、後の DINO の centering + sharpening に受け継がれた。ただし DINO では自己教師あり学習での崩壊防止が目的となり、役割が変化
  - **Mean Teacher の EMA は MixMatch に不要**: SVHN では Mean Teacher の EMA が効くが、MixMatch に追加しても不変または僅かに悪化（表 4）。EMA の効果は文脈依存。BYOL/DINO での EMA は目的が異なる（崩壊防止）
  - **差分プライバシーへの応用が際立つ**: PATE フレームワークとの組み合わせで ε=0.97（VAT: 4.96）という 55 倍のプライバシー改善。ラベル効率の良さがプライバシー保証を直接強化する
  - **新概念ページ [[concepts/semi-supervised-learning]] を作成**: MixMatch は CV wiki で初の半教師あり学習論文。ラベルあり少量 + ラベルなし大量という文脈が今後の ingest で繰り返し現れる可能性があり、独立 concept として早期に確立した
  - 後続候補: FixMatch（高信頼閾値 + RandAugment の MixMatch 後継）、SimSiam（stop-gradient のみで崩壊防止）、MoCo / MoCo v2（momentum encoder 元祖）、UDA（強力なデータ拡張版一貫性正則化）など

## [2026-05-27] ingest | FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence

- 取り込み: `raw/papers/FixMatch- Simplifying Semi-Supervised Learning with Consistency and Confidence.pdf`（PDF 形式）
- 画像対応: ユーザーが手動取得した fig1 を `raw/images/fixmatch_fig1.png` → `raw/assets/fixmatch/fig1.png` に移動（mkdir + mv）。他の図は PDF から取得不可のため省略
- 作成:
  - [[translations/fixmatch]] — Abstract + §1-6 + Broader Impact + Appendix A-E（アルゴリズム擬似コード / 全数値表 + ハイパーパラメータ表 + τ アブレーション / ImageNet 実装詳細 / 拡張手法 Augmentation Anchoring + Distribution Alignment + 他データタイプ応用 / RandAugment + CTAugment + Cutout 変換詳細）の全文和訳（References のみ除外）。Figure 1 を `<figure>` + `<figcaption>` で埋め込み
  - [[sources/fixmatch]] — 初学者向け要約ページ。弱→強の非対称性（役割分担の図解） / τ=0.95 の「量より質」の原則 / 確証バイアスの説明 / 自然なカリキュラム学習 / MixMatch との設計差分 / CIFAR-100 異常（Distribution Alignment 不在の理由）/ weight decay の重要性 / barely supervised 結果 / 用語集
  - [[entities/fixmatch]] — FixMatch エンティティ（アルゴリズム擬似コード / ハイパーパラメータ表 / 全ベンチマーク結果表 / barely supervised 結果 / ImageNet 結果 / τ アブレーション表）
- 更新:
  - [[concepts/semi-supervised-learning]] — frontmatter sources に fixmatch 追加、「FixMatch のポジション」セクション新規追加（弱→強の非対称性 / 確証バイアス / 後継への原則継承 / SSL との思想的接続）、参考セクションに fixmatch 関連ページ追加
  - [[index]] — sources/fixmatch、translations/fixmatch、entities/fixmatch を追加、略称表の FixMatch エントリを [[entities/mixmatch]] → [[entities/fixmatch]] に更新、RandAugment / CTAugment / Cutout / confirmation bias / DA / barely supervised / τ を新規登録
  - [[log]]
- メモ:
  - **「シンプルさによる勝利」**: FixMatch が MixMatch を CIFAR-10/250 ラベルで 11.08% → 5.07% に改善したのは、構成要素を増やしたからではなく減らしたから（MixUp なし、Brier スコアなし、λ_u ウォームアップなし）。「弱→強の非対称性」という単一の設計原則が核心
  - **τ=0.95 で 98% を捨てる**: 疑似ラベルの 98% を捨て残り 2% だけ使う設定が最良という結果は、半教師あり学習の「確証バイアス」問題を正面から解決した。「質 > 量」の原則として後の FlexMatch / FreeMatch / SoftMatch に引き継がれた
  - **CIFAR-100 での逆転**: FixMatch は CIFAR-100 低ラベル条件で ReMixMatch に負ける（48.85% vs 44.28%）。Distribution Alignment（DA）を追加すると 40.14% で逆転する。「FixMatch の設計思想は正しいがクラス多数設定では DA が必須」という重要な例外として記録
  - **弱→強の非対称性と SSL の接続**: FixMatch の「弱い拡張 → 局所的ターゲット、強い拡張 → グローバルな学習」は、DINO の multi-crop（局所 → 大域）や teacher-student の EMA 非対称性と思想的に通底する。SeSL と SSL は独立して発展しているが、「データ拡張による不変性の学習」という点で同じ根を持つことを [[concepts/semi-supervised-learning]] に明記
  - **barely supervised の衝撃**: 1 クラス 1 枚（計 10 ラベル）で 64% 中央値は、「ラベルが極端に少ない現実的シナリオ」での SSL 事前学習なし SeSL の可能性を示す。医療画像・希少疾患等の応用意義が大きい
  - **weight decay の重要性が特記事項**: §5.2 で 1 桁小さい weight decay が CIFAR-10 で +10pt 悪化をもたらすという事実は、実装者が見落としやすいが決定的に重要なハイパーパラメータ設定
  - 後続候補: FlexMatch（動的閾値 FixMatch）/ UDA（強力なデータ拡張版一貫性正則化）/ ReMixMatch（分布アライメント統合）/ SimSiam（stop-gradient のみで崩壊防止）/ MoCo / MoCo v2 など

## [2026-05-27] ingest | FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling

- 取り込み: `raw/papers/FlexMatch_ Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling.md`（ar5iv 由来 markdown）
- ダウンロード: `raw/assets/flexmatch/` に fig1.png（CPL 概念図）、fig2.png（実行時間比較）、fig4a.png（τ アブレーション）の 3 枚を ar5iv URL から保存（markdown 形式のため URL 直接取得可能）
- 作成:
  - [[translations/flexmatch]] — Abstract + §1-6 + Broader Impact + Appendix A（ハイパーパラメータ表 / クラスごと精度 / 中央値エラー率 / Precision/Recall/F1/AUC）+ Appendix B（TorchSSL 概要 / BatchNorm Controller / 全 4 データセットベンチマーク表 8-11）の全文和訳。図1・2・4a を `<figure>` + `<figcaption>` で埋め込み（References のみ除外）
  - [[sources/flexmatch]] — 初学者向け要約ページ。CPL の動的閾値設計（σ_t(c) の直感 / β_t(c) の計算 / ウォームアップ機構 / 凸マッピング関数）/ FixMatch との設計差分表 / SVHN 失敗事例の詳細考察（クラス不均衡 × 簡単タスクの組み合わせ）/ カリキュラム学習思想の接続 / アブレーション 3 要素 / FreeMatch・SoftMatch への流れ
  - [[entities/flexmatch]] — FlexMatch エンティティ（CPL 核心式フロー / 全ベンチマーク結果比較表 / 収束速度比較 / クラスごと精度表 / 後継手法表）
- 更新:
  - [[concepts/semi-supervised-learning]] — frontmatter sources に flexmatch 追加、「FlexMatch のポジション」セクション新規追加（CPL のカリキュラム学習思想 / SVHN 失敗の教訓 / 適応閾値への系譜転換）、参考セクションに flexmatch 関連ページ追加
  - [[entities/fixmatch]] — 後継手法セクション新規追加（FlexMatch / FreeMatch / SoftMatch の比較表）、関連ページに entities/flexmatch 追加
  - [[index]] — sources/flexmatch、translations/flexmatch、entities/flexmatch を追加、略称表に FlexMatch/CPL/σ_t(c)/β_t(c)/カリキュラム学習/TorchSSL/FreeMatch/SoftMatch を登録
  - [[log]]
- メモ:
  - **「固定閾値 → 適応閾値」への転換点**: FlexMatch が 2021 年に「クラスごとに閾値が違って当然」を証明したことで、2022 年以降の SeSL 論文では適応閾値が設計の標準軸になった。FixMatch の「τ=0.95 で 98% 捨てても平気」に対する「でも全クラスに同じを課すのは非合理」という答え
  - **CPL のゼロコスト設計**: 追加パラメータなし・追加推論なしという設計制約は、論文全体を通じて一貫した主張。実装は「損失計算時に予測クラスをカウントするだけ」という単純さで、後の TorchSSL や他手法への統合を容易にした
  - **SVHN 失敗事例の重要性**: 論文が正直に「CPL が裏目に出るケース」を分析している点が貴重。「クラス不均衡 + 簡単タスク」の組み合わせでは FixMatch の固定高閾値の方が安定する。これは CPL の仮定（ラベルなしデータがクラス間でバランス）が崩れるケースとして明示化すべき知識
  - **収束速度の劇的改善**: CIFAR-100/400L で 200K 反復時点（FixMatch: 56.35%、FlexMatch: 94.29%）という差は、単なる最終精度改善を超えた実用的インパクト。特に「早期終了で十分な性能を出したい」場面での価値が大きい
  - **TorchSSL の副産物**: FlexMatch と同時公開された TorchSSL（PyTorch ベース 9 手法統合コードベース）は、SSL 研究の再現性向上と公正な比較を目的としている。BatchNorm Controller など実装上の知見も含む
  - **東工大 × Microsoft という組み合わせ**: SimCLR/FixMatch が Google Research、MixMatch が Google Research、DINOv2 が Meta という中で、FlexMatch は東工大（奥村・篠崎研）と Microsoft Research Asia からの貢献。SeSL 研究の地理的多様性
  - 後続候補: FreeMatch / SoftMatch（閾値適応の発展形）/ UDA（強い拡張版一貫性正則化）/ ReMixMatch（分布アライメント統合）など

## [2026-05-28] ingest | Revisiting Semi-Supervised Learning in the Era of Foundation Models

- 取り込み: `raw/papers/Revisiting Semi-Supervised Learning in the Era of Foundation Models.pdf`（30 ページ PDF、本文のみ、ユーザー指示で Appendix A-C を除外）
- 画像対応: ユーザーが手動取得した 6 枚を `raw/images/fig{1..6}.png` → `raw/assets/revisiting-ssl-foundation-models/fig{1..6}.png` に移動（mkdir + mv）
- 作成:
  - [[translations/revisiting-ssl-foundation-models]] — Abstract + §1-7 + Acknowledgment の本文和訳（References, Appendix A-C は除外）。図 1（Venn 図 + アンサンブル curve）/ 図 2（V-PET 4 フェーズ図）/ 図 3（SSL 精度比較）/ 図 4（エントロピー比較）/ 図 5（ランキング頻度）/ 図 6（スケーリング分析）を `<figure>` で埋め込み。表 1-4 を全て含む
  - [[sources/revisiting-ssl-foundation-models]] — 初学者向け要約。「VFM 上では既存 SSL は Labeled-only PEFT を凌駕できない」という衝撃的発見 / V-PET の 4 フェーズ設計 / Mean Labels が Logits/Probs を凌駕する理由（キャリブレーション問題）/ 7 教師なし基準による公正ハイパラチューニング / VTAB ベース 6 データセット新ベンチマーク / MixMatch→FixMatch→FlexMatch→V-PET の系譜上対比表
  - [[entities/v-pet]] — V-PET アルゴリズム（4 フェーズ擬似コード / ハイパラ表 / 全 12 設定結果表 / Mean Labels の優位性 / 系譜上の対比 / 計算コスト）
  - [[concepts/parameter-efficient-fine-tuning]] — **新規 concept ページ**。PEFT の必要性 / 5 種類の分類（加算型 / 低ランク型 / プロンプト型 / 選択型 / プロジェクション型）/ LoRA・AdaptFormer・VPT・BitFit・ConvPass・Fact-TT の特性比較表 / SeSL × PEFT の出会いと V-PET / CV 領域での発展経緯（2019 Adapter → 2025 V-PET）
- 更新:
  - [[concepts/semi-supervised-learning]] — frontmatter sources に revisiting-ssl-foundation-models 追加、tags に foundation-model / peft 追加、related に foundation-model / parameter-efficient-fine-tuning 追加、「2025 年の転換点：VFM 時代の SeSL の再検討」セクション新規追加（3 つの発見・V-PET 概要・系譜上の位置づけ）、参考に v-pet / PEFT 概念追加
  - [[concepts/foundation-model]] — 「VFM 時代の SeSL への波及」セクション新規追加、関連ページに PEFT / SeSL / V-PET / revisiting-ssl-foundation-models 追加
  - [[entities/fixmatch]] — 「VFM 時代における評価（2025 年）」セクション新規追加（DINOv2/LoRA で FixMatch 47.6% vs Labeled Only 56.0% という大幅悪化を明示）、関連ページに v-pet 追加
  - [[entities/flexmatch]] — 同様に VFM 時代評価セクション追加（DINOv2/LoRA で FlexMatch 48.6% vs Labeled Only 56.0%）
  - [[index]] — sources/translations/entities/concepts の各セクションに追加、略称表に VFM/PEFT/LoRA/AdaptFormer/BitFit/VPT/V-PET/Mean Labels/VTAB/DTD/SUN397/RESISC45/Retinopathy/CLEVR-C/AMI/ARI/V-Measure/FMI/BNM/RankMe/CHI/FineSSL/SoftMatch 等を新規登録（24 項目）
  - [[log]]
- メモ:
  - **スクラッチ前提 → 基盤モデル前提への大転換**: MixMatch (2019) → FixMatch (2020) → FlexMatch (2021) の系譜は全て「Wide ResNet をスクラッチから訓練する」前提だった。本論文はこの前提を VFM 時代に問い直し、「VFM 上では既存 SSL の貢献は驚くほど小さい」を実証。SeSL 研究のパラダイムシフトを宣言した重要研究
  - **DINOv2 上での既存 SSL の悪化**: 表 3 の数値が最も衝撃的。FixMatch/FlexMatch/SoftMatch/FineSSL が DINOv2 ベースで Labeled Only より顕著に悪化（−4 〜 −8pt）。SSL がむしろ汎化能力を破壊している証拠
  - **V-PET のミニマリスト設計**: 「閾値 τ=0、強拡張なし、MixUp なし、1 ラウンド self-training、Mean Labels アンサンブル」と引き算ばかりで FixMatch/FlexMatch を凌駕する。シンプルさの新たな勝利（FixMatch が MixMatch を凌駕した時と同じパターン）
  - **Mean Labels（one-hot 平均）の発見**: Mean Logits / Mean Probabilities ではなく one-hot 化してから平均することで、異なるモデルの「キャリブレーション差」を無効化。図 4 が示すエントロピーギャップが定量的根拠
  - **PEFT が SeSL の主役に**: LoRA / AdaptFormer / BitFit / VPT 等の PEFT 手法群が VFM × SeSL の中心ツールに昇格。新規 concept ページ [[concepts/parameter-efficient-fine-tuning]] を作成して整理。これは将来の VFM 系 ingest で繰り返し参照される基盤概念
  - **公正ハイパラチューニング（7 教師なし基準）**: 過去 SeSL 研究の多くがテストセットでのチューニングという隠れたデータリークを抱えていた問題に対し、AMI/ARI/V-Measure/FMI/BNM/RankMe/CHI の 7 基準を平均ランクで統合するプロトコルを提案。今後の SeSL 評価の標準になる可能性
  - **新ベンチマーク（VTAB ベース）の妥当性**: CIFAR/Food101 では凍結 VFM の線形プローブが既に高精度を出してしまうため、SSL の効果を評価できない。DTD/SUN397/RESISC45/Retinopathy/CLEVR-C/KITTI という「VFM が苦戦する」6 領域を選定したのは説得力ある設計判断
  - **既存 wiki 概念群との強い接続**: 本論文は CLIP/DINOv2（[[entities/clip]]/[[entities/dinov2]]）をバックボーンとし、FixMatch/FlexMatch（[[entities/fixmatch]]/[[entities/flexmatch]]）を主要比較対象とする。wiki の既存 6 ページ以上に直接影響。VFM × SeSL という新しい交差領域を確立
  - **Ohio State University の貢献**: SimCLR/FixMatch（Google）/MixMatch（Google）/FlexMatch（東工大・MSRA）/DINOv2（Meta）に続き、Ohio State から SeSL 研究の主要貢献。研究機関の地理的多様化が継続
  - 後続候補: FineSSL（CLIP × 平衡マージン softmax SeSL）/ V-PETL Bench（PEFT ベンチマーク）/ LoRA 原典 / AdaptFormer 原典 / VPT 原典 / FreeMatch / SoftMatch 原典 / Visual Task Adaptation Benchmark（VTAB）原典 など

## [2026-05-28] ingest | Self-Supervised Learning Powered by Synthetic Data From Diffusion Models: Application to X-Ray Images (I-SynMed)

- 取り込み: `raw/papers/Self-Supervised Learning Powered by Synthetic Data From Diffusion Models_ Application to X-Ray Images.md`（IEEE Access 2025、Web クリップ markdown）
- **特例: 画像取得不可** — IEEE Xplore の画像 URL は認証が必要で、curl で取得を試みた 5 ファイルは全て同サイズ（4141 bytes）のエラーレスポンス。`raw/assets/i-synmed/` ディレクトリは作成せず、翻訳内では原典 URL を参照する形で対応
- 作成:
  - [[translations/i-synmed]] — Abstract（原典 markdown では切れている）+ §1-5 + 謝辞の全文和訳。図 1-9・表 1-7 の数値情報を本文中にテキストで保持（References は除外、独立 Appendix なし）
  - [[sources/i-synmed]] — 初学者向け要約。「DDPM 合成画像 → DINO 事前学習」パイプラインの設計詳細 / 統計検定結果（p=0.107〜0.999、すべて有意差なし）/ なぜ「合成 = 実」が成立するかの仮説 / IEEE Access の位置づけと top-tier ML 会議との質的差の明示 / DDPM の memorization 問題への注意喚起
  - [[entities/i-synmed]] — I-SynMed パイプライン（4 phase 図 / DDPM・DINO のハイパラ / 全ベンチマーク数値表 / アブレーション結果）
  - [[concepts/diffusion-model]] — **新規 concept ページ**。DDPM/Stable Diffusion 系列の俯瞰。順方向・逆方向拡散の数式 / 再パラメータ化トリック / UNet 標準アーキテクチャと DiT への発展 / 系譜図（DPM 2015 → DDPM 2020 → DDIM 2021 → SDE 2021 → LDM 2022 → DiT 2023 → Sora 2024）/ DAE との理論的関係 / SSL × 合成データ事前学習の位置づけ / 限界（推論コスト・memorization 問題）
- 更新:
  - [[concepts/self-supervised-learning]] — frontmatter sources/related に i-synmed / diffusion-model 追加、参考セクションに合成データ×SSL の現代的接続を明示
  - [[concepts/denoising-autoencoder]] — 関連ページに diffusion-model / i-synmed 追加（DAE → DDPM の理論的継承を強化）
  - [[entities/dino]] — 「応用領域：医療画像」セクション新規追加（I-SynMed での DINO+ViT-16 採用、DDPM 合成データとの組み合わせ）、関連ページに i-synmed / diffusion-model 追加
  - [[index]] — sources/translations/entities/concepts の各セクションに追加、略称表に DDPM/Stable Diffusion/DDIM/LDM/DiT/Classifier-free Guidance/score-based SDE/I-SynMed/UNet/FID/IS/SSIM/t-SNE/COVIDx CXR-4/NIH Chest X-ray/SIIM-ACR Pneumothorax/LightlySSL/バイオマーカー/フェデレーテッドラーニング を新規登録（19 項目）
  - [[log]]
- メモ:
  - **「合成 = 実」を統計的に実証した応用研究**: 手法的新規性は低い（DDPM + DINO の組み合わせ）が、医療画像領域での「合成データ事前学習」の実証的根拠として価値がある。複数の下流タスク（肺炎/気胸の分類・セグメンテーション）で統計検定（t-test, Mann-Whitney U, Shapiro-Wilk, FDR 補正）込みで「有意差なし」を確認している点が良質
  - **DDPM の memorization 問題は未検証**: 論文の motivation は「プライバシー保護のための合成データ」だが、Carlini et al. 2023 等が指摘する拡散モデルの訓練データ記憶問題に正面から取り組んでいない。論文自身も「将来研究」と認める重大な制限
  - **新規 concept ページ [[concepts/diffusion-model]] を作成**: wiki に拡散モデルの体系的解説が欠けていたため、本 ingest を機会に新規作成。今後 Stable Diffusion / Sora / ControlNet / Diffusion Transformer 等の ingest で繰り返し参照される基盤概念
  - **DAE → DDPM の理論的継承**: [[concepts/denoising-autoencoder]] 内に既に「拡散モデルとの繋がり」セクションがあったため、専用 concept ページとの相互リンクで補完。「**MAE と拡散モデルは DAE という祖先を共有する**」という見方がさらに明確化
  - **画像取得不可の運用例**: IEEE Xplore は CDN 認証が厳しく、Web Clipper でクリップした markdown 内の画像 URL から curl/wget で取得できない。CLAUDE.md §7 の「ダウンロードできない場合は元 URL をそのまま使う」を実践、翻訳ファイル内で IEEE URL を引用してコメント注釈
  - **IEEE Access の位置づけ**: トップ ML 会議（NeurIPS/ICCV/CVPR）ではなく open access ジャーナル（採択率高め）に投稿されている事実を sources ページ内で明示。wiki 利用者が論文の出自と質を理解するための重要情報
  - **DINO の医療応用事例として価値**: 既存 [[entities/dino]] エンティティの応用領域セクションを追加することで、DINO が医療画像でも有効であることを wiki 内で参照可能にした
  - **ablation の wiki 知識との整合**: DINO+ViT が MoCo+ResNet を大きく上回るという結果は、[[concepts/self-supervised-learning]] の「ViT は SSL と相性が良い」「DINO は自己蒸留として強力」という既存知見と一致
  - **後続候補**: Stable Diffusion（Rombach et al. 2022）/ Latent Diffusion 原典 / DDPM 原典（Ho et al. 2020）/ Classifier-free Guidance 原典 / Score-based SDE / DiT（Diffusion Transformer）/ DreamBooth / ControlNet / Sora / MedSAM / RadFM など。医療画像基盤モデル系統と拡散モデル系統の双方を強化すべき

## [2026-05-28] ingest | EVA-X: a foundation model for general chest x-ray analysis with self-supervised learning

- 取り込み: `raw/papers/EVA-X_ a foundation model for general chest x-ray analysis with self-supervised learning - npj Digital Medicine.md`（npj Digital Medicine 2025、Web クリップ markdown）
- ダウンロード: `raw/assets/eva-x/` に fig1-6.webp を Springer Nature の CDN から取得（一度 fig4 が HTML エラーレスポンスを返したが、再試行で WebP 取得成功）
- 作成:
  - [[translations/eva-x]] — Abstract + §1-4（Methods 含む完全本文）+ Data/Code availability の全文和訳。図 1-6 を `<figure>` で埋め込み、表データ・数式を本文中に保持（References と Acknowledgments と Supplementary は除外）
  - [[sources/eva-x]] — 初学者向け要約。EVA-02 系統の医療版という位置づけ / dual ViT（学習可能 + 凍結 CLIP トークナイザ）の核心設計 / iBOT との詳細対比（online vs frozen external tokenizer）/ I-SynMed との対比表 / 11 下流タスク SOTA の具体的数値 / EVA-X-Ti (6M) が 13× FLOPs の MGCA-B を上回る効率性の意義 / 1% ラベルで COVID-19 精度 95% の意味
  - [[entities/eva-x]] — EVA-X モデルファミリー（Ti/S/B 3 サイズ）スペックシート。アーキテクチャ詳細 / 全実験数値表 / iBOT との対比表 / EVA 系統の系譜
- 更新:
  - [[concepts/self-supervised-learning]] — frontmatter sources に eva-x 追加、参考セクションに eva-x 関連ページ追加
  - [[concepts/masked-image-modeling]] — 「EVA / EVA-02 / EVA-X 系統：凍結 CLIP トークナイザによる MIM」セクション新規追加（EVA 系統が MIM の主要な新潮流であることを明示）
  - [[concepts/online-tokenizer]] — 「Online vs Frozen-External Tokenizer：対比設計」セクション新規追加（iBOT/DINOv2 系の online と EVA/EVA-X 系の frozen external の対立を明示化、両者の補完性を整理）
  - [[concepts/foundation-model]] — 関連ページに eva-x / i-synmed 追加（医療基盤モデルの代表例として）
  - [[entities/ibot]] — 後続セクションに EVA-X を追加（iBOT の online tokenizer を凍結外部 CLIP に置き換えた設計バリアント）
  - [[entities/i-synmed]] — 「EVA-X との対比」セクション新規追加（同じ胸部 X 線 SSL でも対照的な 2 アプローチを明示）
  - [[index]] — sources/translations/entities の各セクションに追加、略称表に EVA-X/EVA/EVA-02/EVA-CLIP/MGCA/MedKLIP/BioViL/GLoRIA/Medical MAE/SelfMedMAE/Merged-520K/CXR14/CheXpert/MIMIC-CXR/Ark+/CXR-Foundation/CheXagent/XrayGPT/dual ViT/frozen external tokenizer/Sub-LN/Deepseek-v3/Grad-CAM/UperNet を新規登録（24 項目）
  - [[log]]
- メモ:
  - **医療基盤モデルの本格構築の代表例**: npj Digital Medicine（Nature 系列）に掲載された質の高い研究。「EVA-02 の設計レシピが医療に転移できる」を実証。I-SynMed（IEEE Access）とは対照的に、システム研究としての規模感と完成度
  - **online vs frozen-external tokenizer という対比軸の確立**: iBOT 系（online、self-evolving）と EVA 系（frozen external pretrained CLIP）の設計対立は MIM 研究の重要な分岐点。新規セクションで [[concepts/online-tokenizer]] に整理。両者は対立しつつ補完的（online はドメイン横断的、frozen は強い既存 CLIP を活かす）
  - **I-SynMed との対比ペア**: 同じ「胸部 X 線 + SSL」でも、データ（合成 vs 実）、SSL アルゴリズム（DINO vs MIM）、ジャーナルの質（IEEE Access vs npj Digital Medicine）、規模感（応用 vs システム）が全く異なる。両 entity ページに相互比較表を配置
  - **EVA-X-Ti (6M) の効率性インパクト**: 13× FLOPs を持つ MGCA-B を上回るという結果は、設計優位性の端的な証拠。「より小さく、より速く、より良い」を達成
  - **1% ラベルで 95% 精度の臨床的意義**: ラベルアノテーションが希少な医療現場では、この効率性が直接的な実用価値を持つ。深層学習の医療普及のボトルネック解消につながる
  - **訓練の安定性が異次元**: EVA-X の標準偏差 0.03 は Medical MAE/MGCA/BioViL 比で 2-5 倍の安定性。臨床実装可能性に直結する重要指標
  - **EVA 系統の wiki 入口**: 今回の ingest で EVA / EVA-02 / EVA-CLIP の名前と概念が wiki に初登場。これらの原典 ingest を将来の優先候補に
  - **Discussion で LLM 統合に言及**: CheXagent / XrayGPT 等の医療 VLM との組み合わせを将来課題として明記。医療画像エージェントの方向性を予示
  - **データバイアスへの自己批判**: 論文自体が Glocker et al. 2023 を引用し「CXR14/CheXpert/MIMIC の異質性とバイアス」を認めている誠実さ。出版倫理の高さを示す
  - **Deepseek-v3 でレポート解析**: 中国の LLM を実世界アノテーションに使用（F1 99% 対医師）。海外の Llama や GPT-4 ではなく国産 LLM を使う中国研究の傾向
  - **後続候補**: EVA / EVA-02 / EVA-CLIP 原典（自然画像基盤モデル系統）/ MGCA 原典（医療 CLIP）/ MAE 系医療版（Medical MAE, SelfMedMAE 原典）/ Ark+（Nature 2025 医療基盤モデル）/ CXR-Foundation (ELIXR) / CheXagent / XrayGPT / MedSAM など。医療基盤モデルと EVA 系統の両方を強化すべき

## [2026-05-27] ingest | A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)

- 取り込み: `raw/papers/A Simple Framework for Contrastive Learning of Visual Representations.md`
- ダウンロード: `raw/assets/simclr/` に fig1.png（性能比較棒グラフ）、fig5.png（拡張アブレーション行列）、fig7.png（モデルスケール比較）、fig9.png（バッチサイズ×エポック数比較）を保存
- 作成:
  - [[translations/simclr]] — Abstract + §1-8 + Appendix A-C の全文和訳（References のみ除外）
  - [[sources/simclr]] — 初学者向け要約ページ（4 コンポーネント設計 / 主要アブレーション / 実験結果 / 限界 / 用語）
  - [[entities/simclr]] — SimCLR モデルエンティティ（4 コンポーネント表 / 主要結果 / 後継手法系統図）
- 更新:
  - [[concepts/contrastive-learning]] — frontmatter sources 追加 + §1「インスタンス識別」内に SimCLR 核心設計サブセクション追加（3 つの発見、射影ヘッドのフロー図、拡張構成の重要性）、関連ページに simclr 追加
  - [[concepts/self-supervised-learning]] — frontmatter sources に simclr 追加
  - [[index]] — sources/simclr、translations/simclr、entities/simclr を追加、略称表に SimCLR/NT-Xent/LARS/MoCo/PIRL/Global BN を登録
- メモ:
  - **「何が効くかを解明した論文」**: SimCLR の貢献は新しい手法の提案というより、既存要素の何が効いていたかを体系的アブレーションで解明したこと。非線形射影ヘッドの発見（+10%以上）が最大のインパクト
  - **二次情報→一次情報昇格**: [[concepts/contrastive-learning]] には以前から SimCLR への言及があったが、源論文を ingest することで一次情報に基づいたより詳細なセクションに置換
  - **既存の contrastive-learning.md との整合**: ページ構造はほぼ既存のまま維持し、Section 1「インスタンス識別」の下に SimCLR のサブセクションを追加する形をとった
  - **新概念ページは作らなかった**: NT-Xent、LARS 等は sources/simclr.md と index 略称表で完結。MoCo のエンティティページも現時点では不要と判断
  - 後続候補: MoCo / BYOL / SimSiam（非対比型 SSL の実装解明系）、SimCLR v2（知識蒸留への発展）など

## [2026-05-27] ingest | Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning (BYOL)

- 取り込み: `raw/papers/Bootstrap Your Own Latent A New Approach to Self-Supervised Learning.md`
- ダウンロード: `raw/assets/byol/` に fig1.png（ImageNet 比較バーチャート）、fig2.png（BYOL アーキテクチャ図）、fig3a.png（バッチサイズ比較）、fig8.png（BYOL スケッチ / 全体アーキテクチャ）を保存
- 作成:
  - [[translations/byol]] — Abstract + §1-6 + Appendix A-B の全文和訳（References のみ除外）
  - [[sources/byol]] — 初学者向け要約ページ（2 ネットワーク構造 / predictor の役割 / 崩壊回避の機構 / アブレーション結果 / SimCLR との詳細比較 / 用語）
  - [[entities/byol]] — BYOL モデルエンティティ（アーキテクチャ表 / 主要結果 / 後継手法系統図）
- 更新:
  - [[concepts/self-supervised-learning]] — frontmatter sources に byol 追加、BYOL の説明文を拡充（predictor + EMA が両方必要、74.3% vs 69.3%）、参考セクションに byol を追加
  - [[index]] — sources/byol、translations/byol、entities/byol を追加、略称表に BYOL/predictor を登録
  - [[log]]
- メモ:
  - **「負例は必須か → 否」への答え**: BYOL の最大の貢献は「predictor + EMA ターゲットの組み合わせで崩壊を防げる」という発見。この結果が SimSiam（EMA も除去）、DINO（cross-entropy + centering）へと続く非対比 SSL の流れを確立した
  - **二次情報→一次情報昇格**: [[concepts/self-supervised-learning]] には以前から BYOL の記述があったが（「predictor + momentum + BN」）、原典 ingest により「predictor + EMA target（どちらか一方では崩壊する）」という正確な知見に更新
  - **τ アブレーションの重要性**: τ=0（直接コピー）→崩壊、τ=1（固定ランダム）→18.8% でも崩壊せず（これが BYOL の着想源）、τ=0.99 が最適という実験結果は直観と理論の双方を補強する重要データ
  - **新概念ページは作らなかった**: predictor / bootstrapping は sources/byol.md と index 略称表で完結。SimSiam（BYOL から EMA も除去）は次の ingest 候補
  - 後続候補: SimSiam（stop-gradient のみで崩壊防止）、MoCo / MoCo v2（momentum encoder 元祖）、DINO 系（BYOL の自己蒸留解釈を完成）、SimCLR v2（知識蒸留）など

## [2026-05-27] ingest | MixMatch: A Holistic Approach to Semi-Supervised Learning

- 取り込み: `raw/papers/MixMatch_ A Holistic Approach to Semi-Supervised Learning.md`
- ダウンロード: `raw/assets/mixmatch/` に fig1.png（ラベル推定図）、fig4.png（SVHN+Extra 比較グラフ）を保存（fig2/fig3 は原典 markdown に URL 未埋め込み）
- 作成:
  - [[translations/mixmatch]] — Abstract + §1-5 + Appendix A-C（記号定義 / CIFAR-10 全数値表 / SVHN 全数値表 / SVHN+Extra 全数値表 / 13 層 ConvNet 結果）の全文和訳（References のみ除外）
  - [[sources/mixmatch]] — 初学者向け要約ページ（ラベル推定 + シャープニング + MixUp の 3 ステップ / アブレーション重要性順位 / 差分プライバシー応用 / SimCLR/BYOL との比較 / 用語）
  - [[entities/mixmatch]] — MixMatch エンティティ（アルゴリズム構成図 / ハイパーパラメータ表 / 主要結果表 / アブレーション表 / 後継手法表）
  - [[concepts/semi-supervised-learning]] — 半教師あり学習の新規 concept ページ（自己教師あり学習との区別、一貫性正則化 / エントロピー最小化 / MixUp の 3 系統、系譜図、SSL 略称衝突問題の明示）
- 更新:
  - [[concepts/self-supervised-learning]] — related に [[semi-supervised-learning]] 追加、参考セクションに「半教師あり学習：SSL 略称の衝突注意」を追加
  - [[index]] — sources/mixmatch、translations/mixmatch、entities/mixmatch、concepts/semi-supervised-learning を追加、略称表に MixMatch/MixUp/Label Guessing/Sharpening/SeSL/Mean Teacher/VAT/FixMatch/Brier スコア/PATE/Wide ResNet/WRN を登録
  - [[log]]
- メモ:
  - **半教師あり vs 自己教師あり**: MixMatch は SSL（Self-Supervised Learning）ではなく SeSL（Semi-Supervised Learning）。wikiで「SSL」という略称を持つ [[concepts/self-supervised-learning]] との混同を避けるため、新規 concept ページ [[concepts/semi-supervised-learning]] を作成して区別を明示した
  - **MixUp が最大の貢献要素**: アブレーション（表 4）でラベルあき・なし横断 MixUp を除去すると 11.08% → 39.11%（+28pt）という最大の劣化。シャープニング（+16pt）、K 回平均（+5pt）より重要
  - **シャープニングは DINO へ**: MixMatch のシャープニング関数（温度 T=0.5）の思想は、後の DINO の centering + sharpening に受け継がれた。ただし DINO では自己教師あり学習での崩壊防止が目的となり、役割が変化
  - **Mean Teacher の EMA は MixMatch に不要**: SVHN では Mean Teacher の EMA が効くが、MixMatch に追加しても不変または僅かに悪化（表 4）。EMA の効果は文脈依存。BYOL/DINO での EMA は目的が異なる（崩壊防止）
  - **差分プライバシーへの応用が際立つ**: PATE フレームワークとの組み合わせで ε=0.97（VAT: 4.96）という 55 倍のプライバシー改善。ラベル効率の良さがプライバシー保証を直接強化する
  - **新概念ページ [[concepts/semi-supervised-learning]] を作成**: MixMatch は CV wiki で初の半教師あり学習論文。ラベルあり少量 + ラベルなし大量という文脈が今後の ingest で繰り返し現れる可能性があり、独立 concept として早期に確立した
  - 後続候補: FixMatch（高信頼閾値 + RandAugment の MixMatch 後継）、SimSiam（stop-gradient のみで崩壊防止）、MoCo / MoCo v2（momentum encoder 元祖）、UDA（強力なデータ拡張版一貫性正則化）など

## [2026-05-27] ingest | FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence

- 取り込み: `raw/papers/FixMatch- Simplifying Semi-Supervised Learning with Consistency and Confidence.pdf`（PDF 形式）
- 画像対応: ユーザーが手動取得した fig1 を `raw/images/fixmatch_fig1.png` → `raw/assets/fixmatch/fig1.png` に移動（mkdir + mv）。他の図は PDF から取得不可のため省略
- 作成:
  - [[translations/fixmatch]] — Abstract + §1-6 + Broader Impact + Appendix A-E（アルゴリズム擬似コード / 全数値表 + ハイパーパラメータ表 + τ アブレーション / ImageNet 実装詳細 / 拡張手法 Augmentation Anchoring + Distribution Alignment + 他データタイプ応用 / RandAugment + CTAugment + Cutout 変換詳細）の全文和訳（References のみ除外）。Figure 1 を `<figure>` + `<figcaption>` で埋め込み
  - [[sources/fixmatch]] — 初学者向け要約ページ。弱→強の非対称性（役割分担の図解） / τ=0.95 の「量より質」の原則 / 確証バイアスの説明 / 自然なカリキュラム学習 / MixMatch との設計差分 / CIFAR-100 異常（Distribution Alignment 不在の理由）/ weight decay の重要性 / barely supervised 結果 / 用語集
  - [[entities/fixmatch]] — FixMatch エンティティ（アルゴリズム擬似コード / ハイパーパラメータ表 / 全ベンチマーク結果表 / barely supervised 結果 / ImageNet 結果 / τ アブレーション表）
- 更新:
  - [[concepts/semi-supervised-learning]] — frontmatter sources に fixmatch 追加、「FixMatch のポジション」セクション新規追加（弱→強の非対称性 / 確証バイアス / 後継への原則継承 / SSL との思想的接続）、参考セクションに fixmatch 関連ページ追加
  - [[index]] — sources/fixmatch、translations/fixmatch、entities/fixmatch を追加、略称表の FixMatch エントリを [[entities/mixmatch]] → [[entities/fixmatch]] に更新、RandAugment / CTAugment / Cutout / confirmation bias / DA / barely supervised / τ を新規登録
  - [[log]]
- メモ:
  - **「シンプルさによる勝利」**: FixMatch が MixMatch を CIFAR-10/250 ラベルで 11.08% → 5.07% に改善したのは、構成要素を増やしたからではなく減らしたから（MixUp なし、Brier スコアなし、λ_u ウォームアップなし）。「弱→強の非対称性」という単一の設計原則が核心
  - **τ=0.95 で 98% を捨てる**: 疑似ラベルの 98% を捨て残り 2% だけ使う設定が最良という結果は、半教師あり学習の「確証バイアス」問題を正面から解決した。「質 > 量」の原則として後の FlexMatch / FreeMatch / SoftMatch に引き継がれた
  - **CIFAR-100 での逆転**: FixMatch は CIFAR-100 低ラベル条件で ReMixMatch に負ける（48.85% vs 44.28%）。Distribution Alignment（DA）を追加すると 40.14% で逆転する。「FixMatch の設計思想は正しいがクラス多数設定では DA が必須」という重要な例外として記録
  - **弱→強の非対称性と SSL の接続**: FixMatch の「弱い拡張 → 局所的ターゲット、強い拡張 → グローバルな学習」は、DINO の multi-crop（局所 → 大域）や teacher-student の EMA 非対称性と思想的に通底する。SeSL と SSL は独立して発展しているが、「データ拡張による不変性の学習」という点で同じ根を持つことを [[concepts/semi-supervised-learning]] に明記
  - **barely supervised の衝撃**: 1 クラス 1 枚（計 10 ラベル）で 64% 中央値は、「ラベルが極端に少ない現実的シナリオ」での SSL 事前学習なし SeSL の可能性を示す。医療画像・希少疾患等の応用意義が大きい
  - **weight decay の重要性が特記事項**: §5.2 で 1 桁小さい weight decay が CIFAR-10 で +10pt 悪化をもたらすという事実は、実装者が見落としやすいが決定的に重要なハイパーパラメータ設定
  - 後続候補: FlexMatch（動的閾値 FixMatch）/ UDA（強力なデータ拡張版一貫性正則化）/ ReMixMatch（分布アライメント統合）/ SimSiam（stop-gradient のみで崩壊防止）/ MoCo / MoCo v2 など

## [2026-05-27] ingest | FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling

- 取り込み: `raw/papers/FlexMatch_ Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling.md`（ar5iv 由来 markdown）
- ダウンロード: `raw/assets/flexmatch/` に fig1.png（CPL 概念図）、fig2.png（実行時間比較）、fig4a.png（τ アブレーション）の 3 枚を ar5iv URL から保存（markdown 形式のため URL 直接取得可能）
- 作成:
  - [[translations/flexmatch]] — Abstract + §1-6 + Broader Impact + Appendix A（ハイパーパラメータ表 / クラスごと精度 / 中央値エラー率 / Precision/Recall/F1/AUC）+ Appendix B（TorchSSL 概要 / BatchNorm Controller / 全 4 データセットベンチマーク表 8-11）の全文和訳。図1・2・4a を `<figure>` + `<figcaption>` で埋め込み（References のみ除外）
  - [[sources/flexmatch]] — 初学者向け要約ページ。CPL の動的閾値設計（σ_t(c) の直感 / β_t(c) の計算 / ウォームアップ機構 / 凸マッピング関数）/ FixMatch との設計差分表 / SVHN 失敗事例の詳細考察（クラス不均衡 × 簡単タスクの組み合わせ）/ カリキュラム学習思想の接続 / アブレーション 3 要素 / FreeMatch・SoftMatch への流れ
  - [[entities/flexmatch]] — FlexMatch エンティティ（CPL 核心式フロー / 全ベンチマーク結果比較表 / 収束速度比較 / クラスごと精度表 / 後継手法表）
- 更新:
  - [[concepts/semi-supervised-learning]] — frontmatter sources に flexmatch 追加、「FlexMatch のポジション」セクション新規追加（CPL のカリキュラム学習思想 / SVHN 失敗の教訓 / 適応閾値への系譜転換）、参考セクションに flexmatch 関連ページ追加
  - [[entities/fixmatch]] — 後継手法セクション新規追加（FlexMatch / FreeMatch / SoftMatch の比較表）、関連ページに entities/flexmatch 追加
  - [[index]] — sources/flexmatch、translations/flexmatch、entities/flexmatch を追加、略称表に FlexMatch/CPL/σ_t(c)/β_t(c)/カリキュラム学習/TorchSSL/FreeMatch/SoftMatch を登録
  - [[log]]
- メモ:
  - **「固定閾値 → 適応閾値」への転換点**: FlexMatch が 2021 年に「クラスごとに閾値が違って当然」を証明したことで、2022 年以降の SeSL 論文では適応閾値が設計の標準軸になった。FixMatch の「τ=0.95 で 98% 捨てても平気」に対する「でも全クラスに同じを課すのは非合理」という答え
  - **CPL のゼロコスト設計**: 追加パラメータなし・追加推論なしという設計制約は、論文全体を通じて一貫した主張。実装は「損失計算時に予測クラスをカウントするだけ」という単純さで、後の TorchSSL や他手法への統合を容易にした
  - **SVHN 失敗事例の重要性**: 論文が正直に「CPL が裏目に出るケース」を分析している点が貴重。「クラス不均衡 + 簡単タスク」の組み合わせでは FixMatch の固定高閾値の方が安定する。これは CPL の仮定（ラベルなしデータがクラス間でバランス）が崩れるケースとして明示化すべき知識
  - **収束速度の劇的改善**: CIFAR-100/400L で 200K 反復時点（FixMatch: 56.35%、FlexMatch: 94.29%）という差は、単なる最終精度改善を超えた実用的インパクト。特に「早期終了で十分な性能を出したい」場面での価値が大きい
  - **TorchSSL の副産物**: FlexMatch と同時公開された TorchSSL（PyTorch ベース 9 手法統合コードベース）は、SSL 研究の再現性向上と公正な比較を目的としている。BatchNorm Controller など実装上の知見も含む
  - **東工大 × Microsoft という組み合わせ**: SimCLR/FixMatch が Google Research、MixMatch が Google Research、DINOv2 が Meta という中で、FlexMatch は東工大（奥村・篠崎研）と Microsoft Research Asia からの貢献。SeSL 研究の地理的多様性
  - 後続候補: FreeMatch / SoftMatch（閾値適応の発展形）/ UDA（強い拡張版一貫性正則化）/ ReMixMatch（分布アライメント統合）など

## [2026-05-28] ingest | Revisiting Semi-Supervised Learning in the Era of Foundation Models

- 取り込み: `raw/papers/Revisiting Semi-Supervised Learning in the Era of Foundation Models.pdf`（30 ページ PDF、本文のみ、ユーザー指示で Appendix A-C を除外）
- 画像対応: ユーザーが手動取得した 6 枚を `raw/images/fig{1..6}.png` → `raw/assets/revisiting-ssl-foundation-models/fig{1..6}.png` に移動（mkdir + mv）
- 作成:
  - [[translations/revisiting-ssl-foundation-models]] — Abstract + §1-7 + Acknowledgment の本文和訳（References, Appendix A-C は除外）。図 1（Venn 図 + アンサンブル curve）/ 図 2（V-PET 4 フェーズ図）/ 図 3（SSL 精度比較）/ 図 4（エントロピー比較）/ 図 5（ランキング頻度）/ 図 6（スケーリング分析）を `<figure>` で埋め込み。表 1-4 を全て含む
  - [[sources/revisiting-ssl-foundation-models]] — 初学者向け要約。「VFM 上では既存 SSL は Labeled-only PEFT を凌駕できない」という衝撃的発見 / V-PET の 4 フェーズ設計 / Mean Labels が Logits/Probs を凌駕する理由（キャリブレーション問題）/ 7 教師なし基準による公正ハイパラチューニング / VTAB ベース 6 データセット新ベンチマーク / MixMatch→FixMatch→FlexMatch→V-PET の系譜上対比表
  - [[entities/v-pet]] — V-PET アルゴリズム（4 フェーズ擬似コード / ハイパラ表 / 全 12 設定結果表 / Mean Labels の優位性 / 系譜上の対比 / 計算コスト）
  - [[concepts/parameter-efficient-fine-tuning]] — **新規 concept ページ**。PEFT の必要性 / 5 種類の分類（加算型 / 低ランク型 / プロンプト型 / 選択型 / プロジェクション型）/ LoRA・AdaptFormer・VPT・BitFit・ConvPass・Fact-TT の特性比較表 / SeSL × PEFT の出会いと V-PET / CV 領域での発展経緯（2019 Adapter → 2025 V-PET）
- 更新:
  - [[concepts/semi-supervised-learning]] — frontmatter sources に revisiting-ssl-foundation-models 追加、tags に foundation-model / peft 追加、related に foundation-model / parameter-efficient-fine-tuning 追加、「2025 年の転換点：VFM 時代の SeSL の再検討」セクション新規追加（3 つの発見・V-PET 概要・系譜上の位置づけ）、参考に v-pet / PEFT 概念追加
  - [[concepts/foundation-model]] — 「VFM 時代の SeSL への波及」セクション新規追加、関連ページに PEFT / SeSL / V-PET / revisiting-ssl-foundation-models 追加
  - [[entities/fixmatch]] — 「VFM 時代における評価（2025 年）」セクション新規追加（DINOv2/LoRA で FixMatch 47.6% vs Labeled Only 56.0% という大幅悪化を明示）、関連ページに v-pet 追加
  - [[entities/flexmatch]] — 同様に VFM 時代評価セクション追加（DINOv2/LoRA で FlexMatch 48.6% vs Labeled Only 56.0%）
  - [[index]] — sources/translations/entities/concepts の各セクションに追加、略称表に VFM/PEFT/LoRA/AdaptFormer/BitFit/VPT/V-PET/Mean Labels/VTAB/DTD/SUN397/RESISC45/Retinopathy/CLEVR-C/AMI/ARI/V-Measure/FMI/BNM/RankMe/CHI/FineSSL/SoftMatch 等を新規登録（24 項目）
  - [[log]]
- メモ:
  - **スクラッチ前提 → 基盤モデル前提への大転換**: MixMatch (2019) → FixMatch (2020) → FlexMatch (2021) の系譜は全て「Wide ResNet をスクラッチから訓練する」前提だった。本論文はこの前提を VFM 時代に問い直し、「VFM 上では既存 SSL の貢献は驚くほど小さい」を実証。SeSL 研究のパラダイムシフトを宣言した重要研究
  - **DINOv2 上での既存 SSL の悪化**: 表 3 の数値が最も衝撃的。FixMatch/FlexMatch/SoftMatch/FineSSL が DINOv2 ベースで Labeled Only より顕著に悪化（−4 〜 −8pt）。SSL がむしろ汎化能力を破壊している証拠
  - **V-PET のミニマリスト設計**: 「閾値 τ=0、強拡張なし、MixUp なし、1 ラウンド self-training、Mean Labels アンサンブル」と引き算ばかりで FixMatch/FlexMatch を凌駕する。シンプルさの新たな勝利（FixMatch が MixMatch を凌駕した時と同じパターン）
  - **Mean Labels（one-hot 平均）の発見**: Mean Logits / Mean Probabilities ではなく one-hot 化してから平均することで、異なるモデルの「キャリブレーション差」を無効化。図 4 が示すエントロピーギャップが定量的根拠
  - **PEFT が SeSL の主役に**: LoRA / AdaptFormer / BitFit / VPT 等の PEFT 手法群が VFM × SeSL の中心ツールに昇格。新規 concept ページ [[concepts/parameter-efficient-fine-tuning]] を作成して整理。これは将来の VFM 系 ingest で繰り返し参照される基盤概念
  - **公正ハイパラチューニング（7 教師なし基準）**: 過去 SeSL 研究の多くがテストセットでのチューニングという隠れたデータリークを抱えていた問題に対し、AMI/ARI/V-Measure/FMI/BNM/RankMe/CHI の 7 基準を平均ランクで統合するプロトコルを提案。今後の SeSL 評価の標準になる可能性
  - **新ベンチマーク（VTAB ベース）の妥当性**: CIFAR/Food101 では凍結 VFM の線形プローブが既に高精度を出してしまうため、SSL の効果を評価できない。DTD/SUN397/RESISC45/Retinopathy/CLEVR-C/KITTI という「VFM が苦戦する」6 領域を選定したのは説得力ある設計判断
  - **既存 wiki 概念群との強い接続**: 本論文は CLIP/DINOv2（[[entities/clip]]/[[entities/dinov2]]）をバックボーンとし、FixMatch/FlexMatch（[[entities/fixmatch]]/[[entities/flexmatch]]）を主要比較対象とする。wiki の既存 6 ページ以上に直接影響。VFM × SeSL という新しい交差領域を確立
  - **Ohio State University の貢献**: SimCLR/FixMatch（Google）/MixMatch（Google）/FlexMatch（東工大・MSRA）/DINOv2（Meta）に続き、Ohio State から SeSL 研究の主要貢献。研究機関の地理的多様化が継続
  - 後続候補: FineSSL（CLIP × 平衡マージン softmax SeSL）/ V-PETL Bench（PEFT ベンチマーク）/ LoRA 原典 / AdaptFormer 原典 / VPT 原典 / FreeMatch / SoftMatch 原典 / Visual Task Adaptation Benchmark（VTAB）原典 など

## [2026-05-28] ingest | Perception Encoder: The best visual embeddings are not at the output of the network

- 取り込み: `raw/papers/Perception Encoder- The best visual embeddings are not at the output of the network.pdf`（54 ページ PDF、本文 pp.1-10 のみ、ユーザー指示で References pp.11-18 と Appendix A-F pp.19-54 を除外）
- 画像対応: ユーザーが手動取得した 10 枚を `raw/images/fig{1..10}.png` → `raw/assets/perception-encoder/fig{1..10}.png` に移動（mkdir + mv）
- 作成:
  - [[translations/perception-encoder]] — Abstract + §1 Introduction + §2 PE: Core（§2.1-2.3）+ §3 General Features in a Contrastive Disguise + §4 PE: Language Alignment + §5 PE: Spatial Alignment + §6 Related Work + §7 Conclusion + Additional Contributors の本文和訳。図 1-10 を `<figure>` で埋め込み（References と Appendix A-F は除外）。表 1-8 をすべて含む
  - [[sources/perception-encoder]] — 初学者向け要約。「中間層に一般特徴量が育つ」の発見 / 3 バリアント設計（PEcore / PElang / PEspatial）/ 9 段階累積アブレーション / 動画データエンジン / alignment tuning の 2 種 / DINOv3・SigLIP 2 との詳細対比 / SAM 3 backbone 採用 / 用語集
  - [[concepts/alignment-tuning]] — **新規 concept ページ**。「中間層に眠る特徴を最終層に引き出す」ファインチューニング戦略の一般化。共通原則 / 言語アラインメント（PElang）と空間アラインメント（PEspatial）の具体例 / 知識蒸留・LoRA・linear probing との対比 / 既存研究のヒント / 限界と展望
- 更新:
  - [[entities/perception-encoder]] — **大幅書き直し**。データ規模の誤り訂正（86B → 5.4B unique pairs + 86B samples seen）/ PEcore 訓練レシピ 9 段階の詳細 / PElang セクション新規追加（Llama3.2 decoder + 2 層 MLP + 70M サンプル）/ PEspatial セクション新規追加（SAM 2.1 mask logits + 自己層 41 の 2 教師蒸留）/ DINOv3 との対比表を更新（PEspatial が dense SOTA を取り戻した）/ 主要結果テーブル（ゼロショット画像・動画 / MLLM / Dense Prediction / SOTA 検出）
  - [[index]] — sources/perception-encoder、translations/perception-encoder、concepts/alignment-tuning を追加、略称表に PE/PEcore/PElang/PEspatial（更新）、alignment tuning、PVD、LAMB、Attention pooling、Progressive resolution、Mask regularization、AIMv2、InternVideo2、MetaCLIP、SAM 2.1 mask logits、DETA、CoDETR、layer 41、layerwise feature analysis を登録（14 項目）
  - [[log]]
- メモ:
  - **「中間層に一般特徴量が育つ」という発見の衝撃**: 純粋な対比学習（CLIP 損失のみ）が、十分にスケールすれば OCR / VQA / grounding / 検出 / 深度 / 追跡すべてに使える特徴を中間層に育てるという発見は、「対比学習 vs キャプション化 vs SSL」の三つ巴を「対比学習だけで十分」という方向に巻き戻す可能性を持つ。SigLIP 2 が「全部入りレシピ」で多目的化したのと対照的に、PE は「ピュアな対比学習＋ alignment tuning」で同じことを達成
  - **既存 entities/perception-encoder.md の誤り訂正**: 以前のエントリには「86B 画像-テキスト対」と書かれていたが、原典より正しくは「5.4B unique pairs を 86B samples seen まで訓練」。86B は累計サンプル数（実質エポック）であって unique pair 数ではない。**一次情報を読まずに二次情報で書いた wiki ページは誤りを内包しやすい** という典型例
  - **DINOv3 との対立軸の再評価**: DINOv3 が「PE は dense で弱い」と主張していたが、本論文の PEspatial G は SAM 2.1 mask logits + 自己層 41 の 2 教師蒸留で dense 予測 SOTA を取り戻す。ただし PEspatial は **SAM 2.1 という supervised teacher への依存** が残るため、DINOv3 側の「完全ラベルフリーで dense SOTA」という主張への完全な反論にはなっていない
  - **alignment tuning の知識蒸留との違い**: 表面上は似た損失（cosine 類似度、特徴量マッチング）を使うが、世界観が逆。蒸留は「持っていないものを受け取る」、alignment は「持っているものを表に出す」。新規 concept ページで明示
  - **progressive resolution の意外な威力**: 9 段階アブレーション（図 5）で、progressive resolution（98→154→224 を 4B サンプルずつ）が COCO 検出（凍結特徴）で +10 mAP という最大の改善をもたらす。「複雑な手法より単純な訓練スケジュール工夫」が効くという CV あるある
  - **動画 → 画像への波及効果**: 22M 動画でのファインチューンが画像性能を +0.6-4.0% 改善する事実は、「動画キャプションの記述密度の高さ」が image-only 学習を上回ることを示唆。今後の画像基盤モデル訓練で動画の活用が標準になる可能性
  - **SAM 3 の backbone 採用の循環的関係**: SAM 2 → PEspatial の教師、PE → SAM 3 の backbone、という循環が成立。Meta 内の foundation model ファミリーが相互蒸留・相互利用で発展する構造が明確に
  - **「対比 SOTA を JFT-3B / WebLI なしで取り戻した」**: PE 論文の最も誇らしげな主張の一つ。Google の独占的データセット（JFT-3B）や WebLI を使わず MetaCLIP キュレーションだけで対比モデル SOTA を実現した点は、研究コミュニティのオープン性に対する重要な意義を持つ
  - **本文のみ ingest の判断**: Appendix A-F は 36 ページ（pp.19-54）と本文の 3 倍以上のボリュームがあり、訓練詳細・追加 ablation・実装詳細・データセット詳細・PE Video Dataset の詳細などを含む。「本文だけで PE の主張は把握できる」というユーザー判断は妥当。追加 ablation や実装詳細が必要になった時点で Appendix 個別 ingest で対応する
  - 後続候補: AIMv2（Apple のキャプション化型事前学習、PE の主要比較相手）/ InternVideo2（動画ネイティブ事前学習）/ MetaCLIP 原典（PE の事前学習データキュレーション手法）/ AM-RADIO（複数 foundation model 蒸留統合）/ LocCa（SigLIP 2 の decoder ベース事前学習）/ V-JEPA 2（動画 SSL）など

## [2026-05-28] lint | PE ingest 後の整合性点検 + 古い記述訂正

- 点検範囲: wiki 全 85 ファイル（concepts 18 / entities 26 / sources 19 / translations 19 / index, log, overview）
- 結果サマリ: 孤立ページ 0、index と filesystem は完全一致。健康度は良好。古い記述（86B pairs 等）と「DINOv3 が PE を dense で圧倒」断言が PE ingest で陳腐化していたため修正
- P0 修正: 「86B pairs」→「5.4B unique pairs / 86B samples seen」の訂正と PE 年（2024 → NeurIPS 2025）の更新
  - 更新ファイル: [[concepts/zero-shot-transfer]], [[concepts/contrastive-learning]], [[sources/clip]], [[sources/siglip]], [[entities/clip]], [[entities/siglip]]（2 ヶ所）
- P0 修正: PE バリアントリストに PElang 追加（以前は "PEcore / PEspatial" の 2 のみ）
  - 更新ファイル: [[index]] (line 92), [[sources/dinov3]] (略称表), [[entities/dinov3]]（並列・競合）
- P0 副次修正: [[sources/dinov3]] の "SAM / SAM v2" → "SAM / SAM 2" 表記揺れ訂正
- P1 修正: 「DINOv3 が PE/SigLIP 2 を dense で圧倒」を「広く優位、ただし PE PEspatial が後に部分奪還、最大スケールでは DINOv3 依然優位」に条件付き化
  - 更新ファイル: [[overview]]（マイルストーン表 2025 行 + PE 行追加）, [[index]] (line 19), [[concepts/self-supervised-learning]] （ハイブリッド型節 + 参考セクション）
- P2 修正: alignment-tuning へのクロスリファレンス追加
  - 更新ファイル frontmatter: [[concepts/contrastive-learning]], [[concepts/knowledge-distillation]], [[concepts/foundation-model]], [[concepts/weakly-supervised-pretraining]] の `related:` に [[alignment-tuning]] を、`sources:` に [[sources/perception-encoder]] を追加
  - [[concepts/knowledge-distillation]] に「KD と Alignment Tuning の対比（世界観が逆）」セクションを新規追加（古典 KD は「持っていないものを受け取る」、alignment tuning は「持っているものを表に出す」）
  - [[concepts/foundation-model]] と [[concepts/weakly-supervised-pretraining]] の PE 説明を本論文に基づき詳細化（NeurIPS 2025、5.4B/86B、3 バリアント、中間層発見、alignment tuning）
- P2 副次修正: [[overview]] の "[[sources]] と [[concepts]] を列挙" を平文化（フォルダ名 wikilink は dangling 扱いになるため）
- 残存事項（**意図的に未修正**）:
  - [[entities/moco]] dangling リンク（[[sources/byol]], [[entities/byol]], [[entities/simclr]] から参照）: スキーマが「未作成ページマーカー」として許容、MoCo は次の ingest 候補
  - concepts 間の bare-slug `related:` 参照（例: `[[self-supervised-learning]]` 形式）: Obsidian は解決可能、既存パターン尊重で現状維持（lint レポート P2 で現状維持推奨）
  - [[entities/sam-2]] frontmatter alias の "SAM v2"（検索用 alias なので保持）
- メモ:
  - **「二次情報依存ページの誤りは原典 ingest で訂正される」という典型例**: 「PE: 86B image-text pairs」は DINOv3 や SAM 3 経由の二次情報で記述されていたが、PE 原典 ingest で「5.4B unique pairs を 86B samples seen まで訓練」が正しいと判明、6 ファイルで連鎖訂正
  - **「圧倒」の表現変化**: DINOv3 ingest 時には「dense で PE を圧倒」と書けたが、PE ingest 後は「広く優位、ただし条件付き」に変化。**断言は最新文献の影響を受ける** ので、強い表現は文献参照とセットで残すべき
  - **alignment tuning が KD と「世界観が逆」**という観察は、PE 論文の重要な認識論的貢献。KD ページに対比セクションを追加することで、両概念の対比が明示化された
  - **bare-slug vs フォルダ込みパス問題**: 旧 concepts/*.md の `related:` は bare-slug（[[self-supervised-learning]]）、新規ページの alignment-tuning は同パターンを踏襲した。一貫性のため新規追加もすべて bare-slug 形式を採用。スキーマ更新の余地あり

## [2026-05-28] ingest | End-to-End Object Detection with Transformers (DETR)

- 取り込み: `raw/papers/End-to-End Object Detection with Transformers.md`（ar5iv 由来 markdown、本文のみ、ユーザー指示で References と Appendix 0.A は除外）
- 画像対応: ar5iv URL（<https://ar5iv.labs.arxiv.org/html/2005.12872/assets/...>）から 8 枚を curl でダウンロードし `raw/assets/detr/fig{1,2,3,4,6,7,8,9}.png/jpg` に保存（fig5 は markdown に画像 URL なし）
- 作成:
  - [[translations/detr]] — Abstract + §1-6（Acknowledgements 含む）の全文和訳。図 1-9 を `<figure>` で埋め込み（fig5 は欠落）、表 1-5 をすべて含む
  - [[sources/detr]] — 初学者向け要約。集合予測パラダイム / Hungarian 二部マッチング / NMS 不要化の仕組み / 並列デコーディング / Encoder-Decoder 分業（インスタンス分離 vs 末端 attention）/ Object queries の自然な特化 / パノプティック拡張 / DETR ファミリー系譜（Deformable / DAB / DN / DINO-detector / DETA / CoDETR / MDETR）/ ViT との並列性 / 用語集（DETR / Hungarian / NMS / GIoU / DC5 / FPN / PQ / things-stuff など）
  - [[entities/detr]] — DETR モデルエンティティ（4 バリアント仕様表 / Transformer 構成 / 主要設計判断表 / アブレーション知見 / DETR ファミリー系譜図 / Foundation Model 時代の検出ヘッド標準としての位置づけ）
  - [[concepts/object-detection]] — **新規 concept ページ**。物体検出タスク全体の俯瞰（5 系統: R-CNN / YOLO / Anchor-free / DETR / Promptable）、DETR のパラダイムシフトの説明、現代 CV foundation model における検出ヘッド標準
- 更新:
  - [[sources/perception-encoder]] / [[entities/perception-encoder]] — DETA と DETR-style decoder の説明に [[sources/detr]] / [[entities/detr]] リンク追加、関連ページに DETR と object-detection 追加
  - [[entities/sam-3]] — DETR decoder の説明に DETR 系統リンク追加、関連ページに DETR と object-detection 追加
  - [[sources/sam-3]] — 検出器説明の DETR ベース表記に DETR 系統リンク追加、用語集の DETR 略称定義に DETR 系統リンク追加
  - [[concepts/foundation-model]] — SAM 3 の DETR detector 説明にリンク追加
  - [[concepts/promptable-segmentation]] — DETR 系の損失標準表記にリンク追加
  - [[index]] — sources/detr、translations/detr、entities/detr、concepts/object-detection を追加。略称表に DETR/set prediction/bipartite matching/Hungarian algorithm/Hungarian loss/NMS/anchor/proposal/object queries/GIoU/DC5/Faster R-CNN/Mask R-CNN/FPN/Panoptic Quality (PQ)/things/stuff/AP_S/AP_M/AP_L/Deformable DETR/DAB-DETR/DN-DETR/DINO-detector/auxiliary decoding losses/AdamW/Xavier init/Focal loss を登録（27 項目）
  - [[log]]
- メモ:
  - **「2020 年の Transformer 革命の双子」**: DETR（2020 May, ECCV 2020）と ViT（2020 Oct, ICLR 2021）は Transformer を CV に持ち込んだ並列研究。DETR は CNN backbone を維持しつつ検出ヘッドを Transformer に置き換える「ハイブリッド型」、ViT は CNN を完全に排除する「pure Transformer 型」。**DETR の方が先**（5 月発表）であり、CV における Transformer の最初の本格的成功と言える
  - **集合予測パラダイムの波及**: DETR のもう 1 つの革新は「物体検出を **集合の最適輸送（optimal transport）問題** に置き直した」こと。これにより (1) hand-designed components（NMS, anchor, proposal）が消える、(2) パノプティック・grounding・tracking への拡張が自然、(3) Transformer の表現力が直接効く、という三重の波及効果が生まれた
  - **既存 wiki への接続が極めて強い**: SAM 3 の検出器 / PE PEspatial の検出 decoder（DETA）/ SAM の損失設計（focal + dice）すべてが DETR の系譜。**wiki 内で DETR への参照が 14 ページに既に存在**していたが、source/entity ページがなく略称表記述だけで支えられていた状態だった。本 ingest で「ハブ」が正しく置かれた
  - **新概念ページ object-detection の必要性**: 既存 wiki には物体検出という中心タスクのページがなく、ViTDet / NMS / IoU 等の用語が散在していた。新規 concept ページで 5 系統の系譜（R-CNN / YOLO / Anchor-free / DETR / Promptable）と DETR のパラダイムシフトの位置づけを整理。今後の検出系研究（Deformable DETR / DINO-detector 等）の ingest で再利用される基盤
  - **NMS 排除のメカニズムの理解**: 「NMS が不要」は表面的には推論時のアルゴリズム選択に見えるが、本質は **訓練レベルで 1 対 1 Hungarian マッチングが「同じ物体に複数スロット活性化」を構造的に許さない** ことから来る。これは推論時の選択ではなく、訓練の設計思想の帰結であることを強調
  - **小物体弱点と Deformable DETR への伏線**: DETR 原論文は AP_S での -5.5 ポイント弱点を明示し、「FPN が Faster R-CNN を救ったように、未来の研究が DETR を救うだろう」と予言。実際 Deformable DETR（同年 10 月、Zhu et al.）がこれを解決。**論文自身が後続研究の道筋を予言した珍しい例**
  - **訓練の特殊性が CV 検出研究の新常識を作った**: 300-500 エポック / AdamW / 補助デコーディング損失 / backbone-transformer 学習率分離は、その後の DETR ファミリーすべての標準になった。DETR 以前の検出研究の常識（SGD + 12 エポック）からの根本的転換
  - 後続候補: Deformable DETR（DETR の sparse attention 改善）、DINO-detector（DETR ファミリー集大成）、MDETR（テキスト条件付き DETR、SAM 3 の前身）、OWL-ViT（ViT + DETR の open-vocab 検出）、DETA（PE PEspatial が採用）など

## [2026-05-28] ingest | DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection

- 取り込み: `raw/papers/DINO_ DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection.md`（ar5iv 由来 markdown、本文のみ、ユーザー指示で References と Appendix 0.A-0.D は除外）
- **名前の衝突への配慮**: 既存 `entities/dino.md`（SSL の DINO, Caron et al., 2021）と完全同名のため、ユーザー指示により slug `dino-detector` で区別。両ページに相互の disambiguation 注記を追加
- 画像対応: ar5iv URL（<https://ar5iv.labs.arxiv.org/html/2203.03605/assets/...>）から 8 枚を curl でダウンロードし `raw/assets/dino-detector/fig{1-8}.png/jpg` に保存
- 作成:
  - [[translations/dino-detector]] — Abstract + §1-5 の全文和訳。図 1-8 を `<figure>` で埋め込み、表 1-4 を含む
  - [[sources/dino-detector]] — 初学者向け要約。3 つの新技法（CDN / Mixed QS / LFT）の詳細、DAB-DETR + DN-DETR + Deformable DETR の系譜整理、SSL DINO との混同問題の徹底説明、PE PEspatial / SAM 3 への波及、用語集（DINO/CDN/Mixed QS/LFT/DAB-DETR/DN-DETR/Deformable DETR/ATD(k)/Objects365/Florence/SwinV2-G 等）
  - [[entities/dino-detector]] — モデル仕様（4 バリアント表）、3 つの新技法詳細、主要結果（12 epoch 49.4 / 24 epoch 51.3 / SwinL 63.3 AP）、ablation 表、DETR ファミリー系譜図、SSL DINO との詳細対比表
- 更新:
  - [[entities/dino]] — **disambiguation 注記を冒頭に追加**（物体検出 DINO との明確な区別）
  - [[entities/detr]] — DINO-detector への参照を新ページにリンク、DETR ファミリー系譜図で詳細注記
  - [[sources/detr]] — DINO-detector への参照を新ページにリンク、SSL DINO との別物注記を強調
  - [[concepts/object-detection]] — DETR ファミリー系譜図に DINO 検出器の詳細を反映
  - [[index]] — sources/dino-detector、translations/dino-detector、entities/dino-detector を追加。略称表に CDN/Mixed Query Selection/Look Forward Twice (LFT)/Look Forward Once (LFO)/DAB-DETR/DN-DETR/dynamic anchor box/denoising training/iterative box refinement/query selection/ATD(k)/Objects365/SwinL/SwinV2-G/Florence/Grounding DINO を新規登録（15 項目）。既存「DINO-detector」エントリを新ページにリンク、重複していた DAB-DETR/DN-DETR の旧エントリを削除して新詳細版に置換
  - [[log]]
- メモ:
  - **DETR ファミリーの集大成**: 2020 DETR → 2020 Deformable DETR → 2022 Jan DAB-DETR → 2022 Mar DN-DETR → **2022 Mar DINO（本論文）** という流れの中で、DINO は前 3 つを統合し新技法 3 つを追加。この後 DETA (2022 Sep) / CoDETR (2023) / Grounding DINO (2023) / MM-Grounding-DINO (2024) と続く。**DETR が「研究用おもちゃ」から「実用 SOTA」に押し上げられた決定的研究**
  - **名前の衝突の典型例**: SSL の DINO（Caron et al., 2021、Meta/Inria）と検出器の DINO（Zhang et al., 2022、HKUST/IDEA）は完全に独立な研究で、命名の由来も別々（**DI**stillation/**NO** labels vs **DI**mproved/de**N**oising/anch**O**r）。コミュニティの混乱を招いているが、文献では文脈で区別される。wiki では slug `dino` vs `dino-detector` で完全分離
  - **3 つの新技法の理解**:
    - CDN（Contrastive DeNoising）: DN-DETR + hard negative。「物体なし」を学ぶ能力の追加。重複予測抑制と小物体改善（+1.3 AP）
    - Mixed Query Selection: 「encoder の top-K は位置として有用だがコンテンツとしては未精錬」という洞察。位置だけ初期化、コンテンツは学習可能のまま
    - Look Forward Twice: Deformable DETR の Look Forward Once（勾配 detach）の対極。後段層の勾配で前段 box 予測を補正
  - **小物体性能の革新**: AP_S +7.5 という改善は、DETR の伝統的弱点の解決という意味で **FPN が Faster R-CNN を救ったのと同じインパクト**。CDN の小物体 anchor 選択能力の向上（ATD(k) 分析で実証）が貢献
  - **既存 wiki ハブが事前整備されていた**: DETR ingest 時に [[entities/detr]] / [[sources/detr]] / [[concepts/object-detection]] / index 略称表で "DINO-detector" として参照済みだったため、本 ingest で完全にハブ化された
  - **PE PEspatial / SAM 3 への波及**: DETA decoder（PE PEspatial が採用、COCO 66.0 box AP）と DAC-DETR / Plain-DETR（SAM 3 が採用）はいずれも DINO 検出器以降の DETR ファミリー技術。**現代の最先端検出ヘッドはすべて DINO 検出器の直接的後継**
  - **COCO SOTA 競争の転換点**: DINO-SwinL（218M）が SwinV2-G（3.0B）を 1/15 のパラメータで上回ったことで、「巨大 backbone + 古典的検出器」から「効率的 backbone + DETR 系検出器」への移行が決定的になった
  - 後続候補: MDETR / Grounding DINO（open-vocab 拡張）、DETA（PE PEspatial の検出 decoder）、CoDETR（協調訓練 SOTA）、Plain-DETR（DINOv3 + COCO 66.1 AP）、Deformable DETR（DETR ファミリーの基礎を完成）など

## [2026-05-28] ingest | Grounded Language-Image Pre-training (GLIP)

- 取り込み: `raw/papers/Grounded Language-Image Pre-training.md`（ar5iv 由来 markdown、本文のみ、ユーザー指示で References と Appendix A-E は除外）
- 画像対応: ar5iv URL（<https://ar5iv.labs.arxiv.org/html/2112.03857/assets/...>）から 6 枚を curl でダウンロードし `raw/assets/glip/fig{1,2,4,5,6,7}.png` に保存（fig3 と fig8 は markdown に画像 URL なし、本文中で参照されているのみ）
- 作成:
  - [[translations/glip]] — Abstract + §1-6（Acknowledgement 含む）の全文和訳。図 1, 2, 4-7 を `<figure>` で埋め込み、表 1-5 をすべて含む。3.1 の数式（grounding loss、target 拡張）、3.2 の X-MHA 数式、3.3 の self-training 詳細を含む
  - [[sources/glip]] — 初学者向け要約。3 つの中核技術（unified formulation / language-aware deep fusion (X-MHA) / self-training scale-up）の詳細、5 バリアント仕様（A/B/C/T/L）、COCO 49.8 AP ゼロショット、LVIS rare カテゴリ革命、Flickr30K phrase grounding SOTA、ODinW ベンチマーク、prompt tuning の物体検出への持ち込み、SAM 3 への精神的継承、用語集
  - [[entities/glip]] — モデル仕様（5 バリアント表、X-MHA 数式、アーキテクチャ詳細）、主要結果（COCO/LVIS/Flickr30K/ODinW 全表）、訓練データ詳細表、open-vocab 検出系統図（GLIP → Grounding DINO → SAM 3）、SAM 3 との比較表
- 更新:
  - [[entities/dino-detector]] — 表 3 の "GLIP" を [[entities/glip|GLIP]] にリンク
  - [[concepts/object-detection]] — DETR ファミリー後に「**§5 Open-Vocabulary 検出**」セクション新規追加（Bansal → ViLD → MDETR → **GLIP** → OWL-ViT → Grounding DINO → MM-Grounding-DINO の系譜）、§6 Promptable 時代に GLIP の精神的継承を明記
  - [[concepts/zero-shot-transfer]] — GLIP を「**物体検出へのゼロショット転移を確立**」した転換点として追加
  - [[concepts/promptable-concept-segmentation]] — GLIP を SAM 3 PCS の精神的祖先として明示（テキスト名詞句で全インスタンス検出という発想を 2021 年に提示）
  - [[sources/sam-3]] — open-vocab 検出器の弱点セクションで GLIP を最初に挙げる
  - [[entities/sam-3]] — 関連ページに [[sources/glip]] / [[entities/glip]] を追加
  - [[index]] — sources/glip、translations/glip、entities/glip を追加。略称表に GLIP / phrase grounding / region-word alignment / X-MHA / late fusion / deep fusion / DyHead / MDETR / ViLD / GoldG / Cap4M-Cap24M / CC-CC12M / SBU / FourODs / ODinW / EgoHands-Pothole-ThermalDogsandPeople / Flickr30K Entities / Visual Genome / GQA / LVIS APr-APc-APf を新規登録（20 項目）。既存 "Grounding DINO" エントリを GLIP リンク付きに更新
  - [[log]]
- メモ:
  - **「CLIP の object-level 拡張」**: GLIP は CLIP（[[entities/clip]]）の画像レベル対比学習を **region レベル** に持ち込んだ最も成功した例。CLIP が「a photo of a {class}」プロンプトで分類を解いたのに対し、GLIP は「person. bicycle. car. ...」プロンプトで検出を解く
  - **検出 = grounding の等価性**: 単一プロンプトで全クラスが収まる場合、検出と grounding は **理論的かつ経験的に完全に等価**（同じ DyHead モデルが両形式で COCO 49.4 AP）。これは PCS（[[concepts/promptable-concept-segmentation]]）のような後続タスクで再利用される基礎的洞察
  - **language-aware deep fusion の重要性**: §5.2 で「物体検出で prompt tuning が full-tuning に並ぶのは GLIP のような deep fusion モデルでのみ」という発見は、PEFT（[[concepts/parameter-efficient-fine-tuning]]）研究にも波及する重要な経験的法則
  - **self-training で grounding をスケール**: 27M grounding データ（うち 24M が web 由来疑似ラベル）は、CLIP の 400M に比べると桁違いに少ない。だが「semantic richness（5840 万のユニーク名詞句）」が量を補う証拠で、後の Grounding DINO / SAM 3 の **データ戦略の祖** となった
  - **GLIP の弱点が SAM 3 で解決された**: GLIP は「文脈なしの絶対的存在判断」が苦手（"vaccine が画像にあるか" を判断する際に強引に位置を出してしまう）。SAM 3 の **presence head** はこれを「全 region に対するグローバルな existence score」として明示的に分離する革新。GLIP ingest により SAM 3 の presence head の意義がより明確に文脈化された
  - **DyHead ベースの限界と Grounding DINO への移行**: GLIP は古典的 DyHead 検出器に依存していたため、後続研究は DETR ベースに移行。Grounding DINO（IDEA, 2023）は GLIP の「検出 = grounding」を DINO 検出器（[[entities/dino-detector]]）の枠組みに移植し、COCO 63.0 AP を達成。**「GLIP × DINO 検出器」という見方が成立**
  - **ODinW ベンチマークの導入**: 13 多様タスク（EgoHands, Pothole, ThermalDogsandPeople 等）で「ゼロショット GLIP-T > 5-shot DyHead-T」「1-shot GLIP-L ≈ fully-supervised DyHead-T」を実証。これは「**1 つの foundation model で多タスクをカバー**」という実用化の道筋を示した最初の本格的成果
  - **既存 wiki ハブが事前整備済み**: DETR / DINO 検出器 ingest 時に略称表で "Grounding DINO" として参照されていたが、その祖の GLIP がまだ wiki になかった。本 ingest で open-vocab 検出系統の **欠けていた基礎ピース** が埋められた
  - **27 ヶ所の Foundation Model 系統への接続**: GLIP は CLIP / DINO 検出器 / SAM 3 / PE / Grounding DINO / OWL-ViT / MDETR / ViLD と多方向に接続している。**open-vocab 検出ハブ** として今後の研究 ingest の出発点になる
  - 後続候補: Grounding DINO 原典（[Liu et al., ECCV 2024]、GLIP × DINO 検出器）/ MDETR（テキスト条件付き DETR の祖、Kamath et al., ICCV 2021）/ OWL-ViT（ViT + DETR 版 open-vocab、Minderer et al., ECCV 2022）/ ViLD（CLIP 蒸留型 open-vocab、Gu et al., ICLR 2022）など

## [2026-05-28] ingest | Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection

- 取り込み: `raw/papers/Grounding DINO_ Marrying DINO with Grounded Pre-Training for Open-Set Object Detection.md`（ar5iv 由来 markdown、本文のみ、ユーザー指示で References と Appendix A-I は除外）
- 画像対応: ar5iv URL（<https://ar5iv.labs.arxiv.org/html/2303.05499/assets/...>）から x1-x5 の 5 枚を curl でダウンロードし `raw/assets/grounding-dino/fig{1-5}.png` に保存
- 作成:
  - [[translations/grounding-dino]] — Abstract + §1-6（Acknowledgement 含む）の全文和訳。図 1-5 を `<figure>` で埋め込み、表 1-7 を含む。Algorithm 1（language-guided query selection の PyTorch コード）を含む
  - [[sources/grounding-dino]] — 初学者向け要約。3 つの中核技術（Tight 3-Phase Fusion / Language-Guided Query Selection / Sub-Sentence Level Text Representation）の詳細、Grounding DINO は **GLIP × DINO 検出器** という統合論文としての位置づけ、SAM 3 への直接の影響、Closed/Open/REC 全 3 設定での評価、用語集
  - [[entities/grounding-dino]] — モデル仕様（T/L 2 バリアント表、X-MHA 数式、アーキテクチャ詳細）、3 つの新技術詳細、主要結果（COCO/LVIS/ODinW/RefCOCO 全表）、訓練データ詳細、ablation 表、DINO 事前学習からの転送実験、open-vocab 検出ファミリー系統図、限界（segmentation 不可、LVIS rare で GLIP に劣る、closed-set で純粋 DINO に -0.9 AP）
- 更新:
  - [[entities/glip]] — Grounding DINO への参照を新ページにリンク（系統図と関連ページ）
  - [[entities/dino-detector]] / [[sources/dino-detector]] — Grounding DINO 言及を新ページにリンク
  - [[sources/glip]] — Grounding DINO への参照を強化、3-phase fusion 等の詳細を追加
  - [[concepts/object-detection]] — §5 Open-Vocab セクションの Grounding DINO 項目を詳細化
  - [[concepts/zero-shot-transfer]] — Grounding DINO エントリを新規追加
  - [[concepts/promptable-concept-segmentation]] — Grounding DINO を SAM 3 の直接の前駆として明示
  - [[sources/sam-3]] / [[entities/sam-3]] — 既存 open-vocab 検出器の比較で Grounding DINO リンクを追加、関連ページに追加
  - [[index]] — sources/grounding-dino、translations/grounding-dino、entities/grounding-dino を追加。略称表に tight fusion / 3-phase fusion / Language-Guided Query Selection / Cross-Modality Decoder / Sub-Sentence Level Text Representation / Word-level Text / Sentence-level Text / REC / RefCOCO/+/g / RefC / open-set object detection / closed-set object detection / OV-DETR / GLIPv2 / DetCLIP / OmDet / MM-Grounding-DINO / DINO-X を新規登録（18 項目）。既存の "Grounding DINO" エントリを新ページにリンク
  - [[log]]
- メモ:
  - **「GLIP × DINO 検出器」の正統な統合**: 第一著者 Shilong Liu と DINO 検出器著者 Hao Zhang, Feng Li は同じ IDEA / HKUST グループ、共著者 Chunyuan Li, Jianwei Yang は GLIP 著者と同じ Microsoft Research。**両研究の著者が直接協力** して作った正統な統合論文。wiki に既にあった 2 つの祖（[[entities/glip]] と [[entities/dino-detector]]）の結合点として位置付けられた
  - **Tight 3-Phase Fusion の概念整理**: 図 2 の「closed-set 検出器を backbone → neck (A) → query init (B) → head (C) の 3 フェーズに分解」という見方は、open-vocab 検出研究全体の整理に有用なメンタルモデル。GLIP は A のみ、OV-DETR は B のみ、MDETR は A+C、Grounding DINO は **A+B+C** という分類が一望できる
  - **「Transformer 検出器が言語との融合に決定的に有利」という洞察**: Faster R-CNN のような古典的検出器は RoI Pooling や RPN という Transformer と相性の悪いモジュールを持つ。一方 DINO 検出器は層ごとに Transformer ブロックで構成されているため、各層に言語 cross-attention を挿入しやすい。これが論文のもう 1 つの重要な貢献
  - **LVIS rare の弱点が興味深い**: 900 queries 固定が長尾分布に不利、GLIP の DyHead は全 grid を使うため rare に強い、という現象。「**DETR 系の固有の弱点**」として記録した。SAM 3 や PE PEspatial も同じ問題を抱える可能性
  - **REC 評価への拡張の意義**: 先行 open-set 検出研究は novel カテゴリのみで評価していた → 「赤いシャツの男性」のような属性付き指示も評価軸に含めるべき、という主張。RefCOCO/+/g を open-set 検出の標準評価に組み込むパラダイムシフト
  - **DINO 事前学習からの転送が有効**: 表 7 で DINO 事前学習からの fine-tune が LVIS で from-scratch を上回る。「**事前学習された強力な closed-set 検出器を open-set 化する道**」を示した実用的知見
  - **Closed-set で純粋 DINO に劣る**: 表 9 で DINO-4scale 49.0 vs Grounding DINO 48.1。**言語側の追加コンポーネントが最適化を難しくしている**。open-set 性能の代償として明示された限界
  - **既存 wiki ハブが事前整備済み**: GLIP と DINO 検出器の ingest 時に Grounding DINO への参照が両ページの「後継」「open-vocab 拡張」セクションに既に置かれていた。本 ingest により、その結合点が正しくハブ化された
  - **SAM 3 への流れ**: Grounding DINO の弱点（segmentation 不可、認識/位置混在、動画なし）を SAM 3 が解決 — マスク生成 + presence head（認識と位置特定の分離）+ SAM 2 tracker。「**Grounding DINO の限界を完全に補完する SAM 3**」という構図が wiki 内で明確化された
  - **MM-Grounding-DINO / DINO-X が後続**: 2024 年以降の OpenMMLab MM-Grounding-DINO と IDEA DINO-X は本論文の直接の発展。将来の ingest 候補
  - 後続候補: MM-Grounding-DINO（OpenMMLab マルチモーダル拡張）/ DINO-X（IDEA、2024）/ MDETR（GLIP の先駆、Kamath et al., ICCV 2021）/ OWL-ViT（Google、Minderer et al., ECCV 2022）/ ViLD（CLIP 蒸留型 open-vocab、Gu et al., ICLR 2022）/ DetCLIP（large-scale captioning, NeurIPS 2022）/ GLIPv2（GLIP + segmentation）など

## [2026-05-28] ingest | YOLO-World: Real-Time Open-Vocabulary Object Detection

- 取り込み: `raw/papers/YOLO-World_ Real-Time Open-Vocabulary Object Detection.md`（ar5iv 由来 markdown、本文のみ、ユーザー指示で References と Appendix A-C は除外）
- 画像対応: ar5iv URL（<https://ar5iv.labs.arxiv.org/html/2401.17270/assets/...>）から x1-x7 の 7 枚を curl でダウンロードし `raw/assets/yolo-world/fig{1-7}.png` に保存
- 作成:
  - [[translations/yolo-world]] — Abstract + §1-5 の全文和訳。図 1-7 を `<figure>` で埋め込み、表 1-8 をすべて含む。RepVL-PAN（T-CSPLayer + I-Pooling Attention）の数式、Region-Text Contrastive Loss、Prompt-Then-Detect パラダイム、CLIP vs BERT ablation 等の核心を含む
  - [[sources/yolo-world]] — 初学者向け要約。5 つの鍵となる技術（RepVL-PAN / Prompt-Then-Detect / Region-Text Contrastive Loss / 疑似ラベリングパイプライン / CLIP text encoder の決定的優位）、GLIP / Grounding DINO との対比表（精度 vs 速度）、用語集
  - [[entities/yolo-world]] — モデル仕様（S/M/L 3 バリアント表、X-MHA 数式、RepVL-PAN 詳細）、主要結果（LVIS / COCO / OVIS 全表）、訓練データ詳細、open-vocab 検出ファミリーの中での位置、限界、関連ページ
- 更新:
  - [[concepts/object-detection]] — §6 Real-Time Open-Vocab 検出セクション新規追加（YOLO-World）、抽象化階段の説明を更新
  - [[concepts/zero-shot-transfer]] — YOLO-World エントリを新規追加
  - [[sources/glip]] — 系統図に YOLO-World を追加（GLIP-L を疑似ラベル teacher として活用する関係を明示）、関連ページに YOLO-World 追加
  - [[entities/glip]] — 「YOLO-World との関係（実は深い）」セクション新規追加、関連ページに YOLO-World 追加
  - [[sources/grounding-dino]] / [[entities/grounding-dino]] — 関連ページに YOLO-World を追加（精度 vs 速度の対照軸として）
  - [[index]] — sources/yolo-world、translations/yolo-world、entities/yolo-world を追加。略称表に YOLO-World / RepVL-PAN / T-CSPLayer / I-Pooling Attention / max-sigmoid attention / Prompt-Then-Detect / online vocabulary / offline vocabulary / re-parameterization / YOLOv8 / CSPLayer / PAN / Darknet / Region-Text Contrastive Loss / DFL / TOOD / Fixed AP / OVIS / DetCLIP / ZSD-YOLO / MMYOLO / ViLD / TensorRT を新規登録（23 項目）。重複していた DetCLIP / ViLD の旧エントリを削除して新詳細版に置換
  - [[log]]
- メモ:
  - **「YOLO + open-vocabulary = リアルタイム open-vocab」の発見**: GLIP（0.12 FPS）/ Grounding DINO（1.5 FPS）/ DetCLIP（2.3 FPS）の精度志向に対し、YOLO-World（52 FPS）は **2-3 桁高速化** しつつ AP も上回るという稀有な結果。「Pareto フロンティアを根本的に押し出した」研究
  - **Prompt-Then-Detect パラダイムの革新性**: GLIP / Grounding DINO は「画像とテキストを毎フレーム両方エンコード」していたが、YOLO-World は「テキストを事前エンコード → モデル重みに re-parameterize → 推論時はテキスト encoder を削除」。これにより **YOLOv8 と同じ推論コスト** で open-vocab 検出を実現。**実用デプロイメントの本質的解決**
  - **CLIP text encoder への原点回帰**: GLIP（BERT）/ Grounding DINO（BERT）と異なり、YOLO-World は CLIP（自前訓練）のテキスト塔を流用。ablation で BERT 14.6 → CLIP 22.4 AP（APr で +10.1）という劇的差。**「open-vocab 検出の鍵は CLIP の image-text 整合性」** という GLIP からの洞察更新
  - **GLIP の self-training パイプラインの実質的継承**: YOLO-World は GLIP-L を疑似ラベル生成 teacher として活用（CC3M に 246K 画像、821K 注釈）。GLIP → YOLO-World への知識転移は「**高精度 foundation model から軽量実用モデルへの間接的蒸留**」という実用パターンの典型例。本 ingest で wiki 内に GLIP の self-training → YOLO-World の継承関係が明示された
  - **RepVL-PAN の軽量設計の妙**: T-CSPLayer の max-sigmoid attention は softmax の代わりに max を使うことで「カテゴリ間の競合を避ける」（検出では複数カテゴリ同時存在しうる）。I-Pooling Attention は **わずか 27 patch tokens** で画像をテキスト側に注入。両者とも **計算コスト極小** だが効果的
  - **YOLO 系統での open-vocab 化の意義**: YOLOv1（2016）以来 closed-set リアルタイム標準だった YOLO を、YOLO-World が初めて open-vocab 化。**「YOLOv1-v8 → YOLO-World」という YOLO 系の自然な発展軸** が wiki に追加された
  - **「精度志向 vs 実用志向」という直交軸**: 既存 wiki が積み上げてきた「DETR ファミリー」「GLIP ファミリー」は精度志向だったが、YOLO-World は **実用志向の対抗路線**。両軸の存在を [[concepts/object-detection]] で明示
  - **エッジデバイス応用への道**: 自律走行、ロボティクス、AR/VR、警備カメラ等の応用では 30 FPS 前提が一般的。GLIP の 0.12 FPS では 1 フレーム検出に 8 秒、これでは実用にならなかった。YOLO-World の 52 FPS で **エッジ open-vocab 検出が初めて現実的に**
  - **既存 wiki ハブが事前整備されていない**: GLIP / Grounding DINO ingest 時には YOLO-World への参照は wiki に 0 件だった（YOLO-World は CVPR 2024 でかなり新しい論文）。本 ingest により、open-vocab 検出系統に「**実用志向ハブ**」が追加された
  - **概念的に CLIP に最も近い**: GLIP / Grounding DINO が CLIP の精神（image-text contrastive）を拡張しつつ BERT を使用、Grounding DINO は CLIP を使う方向に戻りつつあったが、YOLO-World は **CLIP text encoder を完全流用** することで「CLIP の正統な object-level 拡張」という側面を最も強く持つ
  - 後続候補: MM-Grounding-DINO（OpenMMLab マルチモーダル拡張）/ DINO-X（IDEA、2024）/ YOLO-Worldv2 / OWL-ViTv2 / GLIPv2（segmentation 拡張）/ MDETR（GLIP の先駆）/ DetCLIP（YOLO-World が比較）など

## [2026-05-28] ingest | Grounding DINO 1.5: Advance the "Edge" of Open-Set Object Detection

- 取り込み: `raw/papers/Grounding DINO 1.5_ Advance the "Edge" of Open-Set Object Detection.md`（ar5iv 由来 markdown、本文のみ、ユーザー指示で References と §7 Appendix は除外）
- 画像対応: ar5iv URL（<https://ar5iv.labs.arxiv.org/html/2405.10300/assets/...>）から x1-x16 の 16 枚を curl でダウンロードし `raw/assets/grounding-dino-1-5/fig{1-16}.png` に保存
- 作成:
  - [[translations/grounding-dino-1-5]] — Abstract + §1-6 全文和訳。図 1-16 を `<figure>` で埋め込み、表 1-5 をすべて含む。Pro と Edge 両モデルのアーキテクチャ詳細、Grounding-20M データセット、early/late fusion 比較、COCO/LVIS/ODinW の全結果、定性的可視化（一般物体、長尾、短/長キャプション grounding、密な物体、動画、Side-by-side 比較、エッジデバイス）を含む
  - [[sources/grounding-dino-1-5]] — 初学者向け要約。Pro/Edge 双子戦略、4 つの改善点（ViT-L、Grounding-20M、early fusion + 負例サンプリング、Efficient Feature Enhancer）、YOLO-World との直接対比、SAM 3 / PE との関係、用語集
  - [[entities/grounding-dino-1-5]] — Pro と Edge 両モデル仕様、Efficient Feature Enhancer の 3 段階効率化（P5 のみ融合 + vanilla self-attn + cross-scale fusion）、主要結果（COCO/LVIS/ODinW/Edge 全表）、アーキテクチャ比較（vs Grounding DINO、vs YOLO-World）、限界、関連ページ
- 更新:
  - [[entities/grounding-dino]] / [[sources/grounding-dino]] — Grounding DINO 1.5 への参照を「直接の後継」として追加
  - [[entities/yolo-world]] / [[sources/yolo-world]] — Grounding DINO 1.5 Edge を real-time open-vocab 路線の直接的競合として明示
  - [[concepts/object-detection]] — §5 Open-Vocab セクションに Grounding DINO 1.5 を追加（Pro/Edge 両方の系統図）、§6 Real-Time Open-Vocab に Grounding DINO 1.5 Edge を追加（YOLO-World との双璧として）
  - [[concepts/zero-shot-transfer]] — Grounding DINO 1.5 エントリ新規追加
  - [[index]] — sources/grounding-dino-1-5、translations/grounding-dino-1-5、entities/grounding-dino-1-5 を追加。略称表に Grounding DINO 1.5 / Pro / Edge / Grounding-20M / Efficient Feature Enhancer / EfficientViT-L1 / early fusion / late fusion / hallucination / negative sampling / Lite-DETR / Cross-scale feature fusion / NVIDIA Orin NX / language cache / ODinW35 / DetCLIPv2 / DetCLIPv3 / T-Rex2 / APE / GLEE-Pro / OmDet-Turbo / OWL-ST / MQ-GLIP / V3Det / GranuCap50M / WebLI / Cap24M / fixed AP / API mode を新規登録（29 項目）。既存「TensorRT」を更新
  - [[log]]
- メモ:
  - **「精度と速度の両極を 1 モデルスイートで統合」**: 既存 wiki が分けていた「GLIP/Grounding DINO 系（精度）」と「YOLO-World 系（速度）」を、Grounding DINO 1.5 は **Pro/Edge の双子スイート** という形で同時に実現。**「open-vocab 検出の精度上限と実用下限の両方を同じ研究グループが追求」** という戦略的アプローチが wiki 内で明示化された
  - **論文タイトルの "Edge" の二重の意味**: (1) **分野の先端（"advance the edge"）**、(2) **エッジコンピューティング（edge devices）**。著者陣の意図が明確に伝わる秀逸な命名
  - **IDEA Research の中央拠点化**: DINO 検出器（2023） → Grounding DINO（2024 ECCV） → Grounding DINO 1.5（2024 May） → DINO-X（2024）と、**IDEA が open-vocab 検出の中心的開発拠点として確立**。本 ingest で IDEA の系譜が wiki 内で明確に
  - **Grounding-20M データセットの戦略的位置**: 20M+ grounding 画像（公開ソース）を IDEA 独自のデータエンジンと注釈パイプラインで選別。元の Grounding DINO の 27M（FourODs+GoldG+Cap4M+COCO+RefC）から「**質を上げた選別データ**」へシフト。**「データの量より質」** の典型例
  - **Efficient Feature Enhancer の 3 段階設計の意義**: (1) P5 のみ cross-modality 融合（Lite-DETR の知見）、(2) Deformable → Vanilla self-attn（エッジ GPU 実装容易性）、(3) Cross-scale feature fusion で P3/P4 統合（小物体検出維持）。**3 つの最適化を組み合わせて初めて Orin NX 10 FPS を達成**。エッジデバイス向け Transformer 系の設計指針として参考価値高い
  - **Early/Late fusion のトレードオフ分析**: 論文 §2.1.1 で early fusion = 高再現率 + 高 box 精度 + 高 hallucination、late fusion = 低 hallucination + 低再現率 という明示的な議論。**Pro は early を保持しつつ負例サンプリング戦略改善で hallucination を緩和**。この分析は他の open-vocab 検出研究の設計選択にも応用可能
  - **YOLO-World との直接対比**: Edge モデル（LVIS-mv 36.2 AP, Orin NX 10.7 FPS）と YOLO-Worldv2-L（32.9 AP, 同等の速度）が **互角の競争**。GD 1.5 Edge は EfficientViT + efficient enhancer（Transformer 路線）、YOLO-World は YOLOv8 + RepVL-PAN（CNN 路線）という **異なる最適化アプローチが同じ実用性能に収束**
  - **DetCLIPv3 への +6.9 AP 改善の意義**: LVIS-mv で DetCLIPv3 (Swin-L, 48.8 AP) → GD 1.5 Pro (ViT-L, 55.7 AP) は backbone 差もあるが、**「ViT-L + 質的に選別された 20M データ」の組み合わせが大規模・多様な訓練データより効果的** という重要な経験的知見
  - **API mode の戦略的選択**: モデル重みは API 経由のみで公開（DeepDataSpace プラットフォーム）。完全オープンソースではない点が学術コミュニティから批判されやすいが、**IDEA Research の商用化戦略の一環**。Grounding-20M の詳細が非公開なのも同様の理由
  - **新ベンチマーク ODinW35 の導入**: GLIP/Grounding DINO 系の ODinW（13 datasets）を **35 datasets に拡張**。Grounding DINO 1.5 Pro ZS 30.2 / fine-tune 70.6 という巨大ジャンプ（+40.4）は、**open-vocab 検出の汎化能力の真の限界はまだ遠い** ことを示唆
  - **既存 wiki ハブとの自然な統合**: Grounding DINO ingest 時に「DINO-X」「MM-Grounding-DINO」が後継候補として記載されていたが、Grounding DINO 1.5 は明示的後続として未記載だった。本 ingest により、**Grounding DINO → Grounding DINO 1.5 → DINO-X** という IDEA Research 内部の系譜が完成
  - 後続候補: DINO-X（IDEA、2024 後半）/ MM-Grounding-DINO（OpenMMLab、2024）/ Grounded SAM（IDEA、2024）/ DetCLIPv3（NeurIPS 2024）/ T-Rex2（visual prompt）/ APE（CVPR 2024）/ GLEE-Pro（CVPR 2024）/ OmDet-Turbo（real-time、language cache）など

## [2026-05-28] ingest | DINO-X: A Unified Vision Model for Open-World Object Detection and Understanding

- 取り込み: `raw/papers/DINO-X_ A Unified Vision Model for Open-World Object Detection and Understanding.md`（ar5iv 由来 markdown、本文のみ、ユーザー指示で References は除外）
- 画像対応: ar5iv URL（<https://ar5iv.labs.arxiv.org/html/2411.14347/assets/...>）から x1-x11 の 11 枚を curl でダウンロードし `raw/assets/dino-x/fig{1-11}.png` に保存
- 作成:
  - [[translations/dino-x]] — Abstract + §1-7 全文和訳。図 1-11 を `<figure>` で埋め込み、表 1-7 をすべて含む。Pro と Edge 両モデル、4 種ヘッド（Box/Mask/Keypoint/Language）、3 種プロンプト、Grounding-100M データセット、2 段階訓練戦略、Knowledge Distillation、FP16 量子化、prompt-free 検出、Side-by-side 比較を含む
  - [[sources/dino-x]] — 初学者向け要約。「Grounding DINO 系統の到達点 = unified perception model」の位置づけ、5 つの鍵（3 種プロンプト/4 種ヘッド/Grounding-100M/2 段階訓練/Edge の三重最適化）、SAM 3 との並行進化、YOLO-World との対比、Grounded SAM の代替、用語集（DINO-X / Grounding-100M / Universal Object Prompt / prompt-free / OPT-125M / EfficientViT-L2 / Knowledge Distillation 等）
  - [[entities/dino-x]] — Pro と Edge 両モデル仕様、3 種プロンプトと 4 種ヘッドの詳細、主要結果（COCO/LVIS/Few-shot Counting/Keypoint/Recognition/Captioning/Edge 全表）、アーキテクチャ比較（vs GD 1.5、vs YOLO-World、vs SAM 3）、限界、関連ページ
- 更新:
  - [[entities/grounding-dino-1-5]] / [[sources/grounding-dino-1-5]] — DINO-X への参照を「直接の後継」として追加
  - [[entities/grounding-dino]] / [[sources/grounding-dino]] — DINO-X を「IDEA 系譜の現時点最新版」として追加
  - [[entities/yolo-world]] / [[sources/yolo-world]] — DINO-X Edge を「YOLO-Worldv2-L を精度・速度両方で凌駕」として追加
  - [[entities/sam-3]] — DINO-X を「SAM 3 と並行進化する unified perception 路線」として追加
  - [[concepts/object-detection]] — §5 Open-Vocab と §6 Real-Time セクションに DINO-X を追加（4 ヘッド統合と Universal Object Prompt の革新性を強調）
  - [[concepts/zero-shot-transfer]] — DINO-X エントリ新規追加
  - [[index]] — sources/dino-x、translations/dino-x、entities/dino-x を追加。略称表に DINO-X / Pro / Edge / Grounding-100M / Grounding DINO 1.6 / Universal Object Prompt / prompt-free detection / prompt-tuning / Customized Prompt / Visual Prompt / object tokens / task tokens / RoIAlign / Mask2Former / Mask DINO / ED-Pose / OPT-125M / feature-based distillation / response-based distillation / FP16 量子化 / EfficientViT-L2 / T-Rex2 / Grounded SAM / Grounded SAM 2 / Osprey / GPT4RoI 等 / RefCOCOg / PACO / SS / S-IoU / CIDEr / METEOR / FSC147 / FSCD-LVIS / HInt / PCK@0.05 / OKS / CrowdPose / Human-Art / HaMeR / Hamba を新規登録（37 項目）。既存 API mode を更新
  - [[log]]
- メモ:
  - **「Grounding DINO 系統の到達点」**: IDEA Research の DINO 検出器（2023）→ Grounding DINO（2024 ECCV）→ Grounding DINO 1.5（2024 May）→ Grounding DINO 1.6（中間版、論文なし）→ **DINO-X（2024 Nov）** の系譜で、**box 出力単一タスクから unified object-centric perception model への進化を完了**。本 ingest により、IDEA Research の open-set 検出系譜が wiki 内で **5 世代** にわたって追跡可能になった
  - **論文タイトルの "Unified" の意味**: 3 種プロンプト（Text/Visual/Customized）+ 4 種 head（Box/Mask/Keypoint/Language）で、Grounded SAM のような multi-step pipeline を **単一モデル** に統合。「open-world detection と understanding を統合」という野心
  - **CLIP text encoder への切り替えの意義**: GD 1.5 まで BERT を使用していた IDEA が、DINO-X で **CLIP に切り替え**（[[entities/yolo-world]] と同じ流れ）。**「open-vocab 検出の鍵は CLIP の image-text 整合性」** という観察を IDEA も追認、wiki 内で 2 つの異なる系統（YOLO 系と DINO 検出器系）が同じ結論に達したことが明示化
  - **Universal Object Prompt の革新性**: GLIP/Grounding DINO/GD 1.5/YOLO-World はすべて「テキストプロンプト必須」。DINO-X は **prompt-tuning で「すべての物体」を表す固定 embedding を学習** → 推論時にプロンプトなしで全物体検出可能。**「画面解析、汎用カメラ、対話的でない応用」** に重要、open-vocab 検出の応用領域を拡張
  - **軽量 language head（OPT-125M）の驚異的性能**: Osprey の Vicuna-7B（7B）を **1/56 のサイズで上回る**（LVIS-val SS 71.25 vs 65.24）。これは **「強力な visual backbone + 軽量 language head」** が region understanding の理想構成であることを実証。MLLM の常識（7B-70B が必要）への反論
  - **長尾検出の劇的改善**: LVIS-mv APr 63.3 は GD 1.6 Pro の 57.5 から +5.8 AP、GD 1.5 Pro の 56.1 から +7.2 AP。**Grounding-100M（GD 1.5 の 5×）の効果が長尾で集中的に発現**。「データ量が長尾で効く」という古典的観察の現代 open-vocab 文脈での再確認
  - **DINO-X Edge の三重最適化**: (1) CLIP text encoder（BERT から）、(2) Pro→Edge の Knowledge Distillation（feature + response based）、(3) FP16 量子化（浮動小数点乗算正規化）。Orin NX で **20.1 FPS**（GD 1.5 Edge 10.7 から +87%）かつ LVIS-mv 48.3 AP（YOLO-Worldv2-L 33.0 を **+15.3 AP** 凌駕）。「精度志向と実用志向の両立」という GD 1.5 の戦略を **次の段階に押し進めた**
  - **SAM 3 との並行進化**: DINO-X（2024 Nov）と SAM 3（2025）はそれぞれ IDEA Research と Meta Superintelligence Labs から独立に **unified perception model を目指して開発**。**設計選択の違い**: DINO-X = multi-head + multi-prompt 統合（より広い）、SAM 3 = presence head + 対話性 + 動画（より深い特化）。wiki 内でこの **「並行進化する 2 つの unified perception アプローチ」** が明示化された
  - **Grounded SAM の代替**: Grounded SAM（Grounding DINO + SAM）は IDEA Research の人気パイプライン（2024）だったが、DINO-X は **検出 + マスク + キャプションを単一モデル** に統合。マスク精度では Grounded SAM (SAM-Huge) に劣る（LVIS-mv 43.8 vs 47.7）が、**統一モデルの効率優位**。「pipeline → unified model」への移行を IDEA 自身が実証
  - **MLLM 連携への応用提案**: 論文は「MLLM の hallucination 削減」を重要な動機として挙げる。**object-level な認識能力で MLLM の応答信頼性を高める** という応用提案は、PE / SigLIP 2 等の vision tower 系研究と異なる文脈で MLLM 接続の道筋を示す
  - **既存 wiki ハブとの自然な統合**: GD / GD 1.5 ingest 時に「DINO-X」「MM-Grounding-DINO」が後続候補として記載されていた。本 ingest により、IDEA Research 系譜の **現時点最新版** が wiki に追加され、open-vocab 検出系統が **5 世代の系譜** として完成
  - 後続候補: Grounded SAM 原典（IDEA, 2024）/ T-Rex2 原典（DINO-X の visual prompt 元）/ Mask2Former / Mask DINO（DINO-X mask head 元）/ ED-Pose（DINO-X keypoint head 元）/ Osprey（DINO-X language head の比較相手）/ MM-Grounding-DINO（OpenMMLab、2024）/ DetCLIPv3（NeurIPS 2024）/ APE（CVPR 2024）/ GLEE-Pro（CVPR 2024）など

## [2026-05-28] ingest | InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks

- 取り込み: `raw/papers/InternVL_ Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks.md`（ar5iv 由来 markdown、本文のみ、ユーザー指示で References と Appendix A.1-A.5 は除外）
- 画像対応: ar5iv URL（<https://ar5iv.labs.arxiv.org/html/2312.14238/assets/x1-x4.png>）から 4 枚を curl でダウンロードし `raw/assets/internvl/fig{1-4}.png` に保存（x5 は Appendix 内のため除外）
- 作成:
  - [[translations/internvl]] — Abstract + §1-5 + Acknowledgement の全文和訳。図 1-4 を `<figure>` で埋め込み、表 1-12 を含む（表 7 のフル数値マトリクスは概略のみ）。InternViT-6B のハイパーパラメータ探索（variant 1-6）、QLLaMA の 96 query + cross-attn 設計、3 段階訓練（contrastive 4.98B → ITC/ITM/ITG 1.03B → SFT 4M）、InternVL-C/G/Chat 4 モードを含む
  - [[sources/internvl]] — 初学者向け要約。「6B 視覚 + 8B 言語ミドルウェアで LLM と整列させた初の本格的視覚言語基盤モデル」の位置づけ、3 つの主要設計（パラメータ均衡 / 整合表現 / 段階的整列）、QFormer 比較（188M → 8B の 42×）、視覚知覚 + 視覚言語 + マルチモーダル対話の全結果、限界（訓練コスト・LLaMA 依存）、用語集（InternViT-6B / QLLaMA / InternVL-C/G/Chat / VLLM / QFormer / ITC/ITM/ITG / LAION-COCO / COYO / Wukong / ViT-22B 等を網羅）
  - [[entities/internvl]] — InternViT-6B + QLLaMA + InternVL-C/G/Chat 4 モードの仕様、3 段階訓練、視覚知覚 + 視覚言語 + マルチモーダル対話の全結果、vs CLIP / vs BLIP-2 / vs LLaVA-1.5 / vs ViT-22B の 4 アーキテクチャ比較、公開モデルカタログ（HuggingFace `OpenGVLab/`）、系譜上の位置（CLIP → InternVL → InternVL 1.5/2.0/2.5/3）
- 更新:
  - [[concepts/foundation-model]] — §A WSL 系セクションに InternVL を追加（「視覚エンコーダを 6B、言語ミドルウェアを 8B にスケールアップした初の本格的試み、ViT-22B を 1/3.7 で +12.6 mIoU 上回る」）
  - [[concepts/weakly-supervised-pretraining]] — §A CLIP 系セクションに InternVL を追加（「SigLIP 2 / PE が対比学習自体を改良する路線なのに対し、InternVL は対比 + 生成 + LLM 接続を 1 モデルで統合し、glue layer を 188M → 8B にスケール」）
  - [[concepts/zero-shot-transfer]] — InternVL エントリ新規追加（「視覚エンコーダを 6B + 8B QLLaMA で 1 モデル統合、多言語 + 動画でも頑健にゼロショット転移」）+ 関連ページに [[entities/internvl]] 追加
  - [[entities/clip]] — マルチモーダル AI への発展セクションに InternVL を追加（「CLIP の対比学習を 6B 視覚 + 8B 言語ミドルウェアにスケールアップ。CLIP の対比 + zero-shot を 14B 規模で完成形に」）
  - [[index]] — sources/internvl、translations/internvl、entities/internvl を追加。略称表に InternVL / InternViT-6B / QLLaMA / InternVL-C/G/Chat / QFormer / ITC/ITM/ITG / LLaMA / Vicuna / InternLM / BLIP / BLIP-2 / InstructBLIP / LLaVA / LLaVA-1.5 / MME / POPE / Tiny LVLM / VQAv2 / GQA / VizWiz / TextVQA / NoCaps / OK-VQA / IconQA / AI2D / OCR-VQA / ChartQA / DocVQA / InfoVQA / ST-VQA / LLaVAR / LAION-COCO / COYO / Wukong / CC3M / CC12M / SBU / LAION-en / LAION-multi / ViT-22B / ViT-G / ViT-e / EVA-02-ViT-E / ViT-6.5B / CoCa / LiT-22B / Qwen-VL / Flamingo / IDEFICS / KOSMOS-2 / Shikra / Emu / DreamLLM / PaLI-X-55B / glue layer / VLLM / AGI / JFT-3B を新規登録（30+ 項目）
  - [[overview]] — VLM セクションに InternVL を追加、Foundation Model セクションに「14B クラス対比 + 生成統合の起点」として明示、CV 基盤モデル 3 大系統の WSL 内で位置づけ
  - [[log]]
- メモ:
  - **「初の本格的 6B 視覚 + 8B 言語ミドルウェア」**: 論文は CLIP-L (304M) → InternViT-6B (5.9B) という **19 倍視覚スケール** と、QFormer (188M, BLIP-2) → QLLaMA (8B) という **42 倍 glue scale** を同時に実現した最初の論文。後の InternVL 1.5 / 2.0 / 2.5 / 3 シリーズの直接の起点
  - **3 段階訓練の意義**: CLIP 流の対比訓練（Stage 1 = 4.98B 対）で頑健な視覚表現を獲得し、BLIP-2 流の ITC/ITM/ITG（Stage 2 = 1.03B）で生成能力を獲得し、LLaVA 流の SFT（Stage 3 = 4M）で対話化。**「過去 5 年の VLM 流派をすべて取り込んだ統合レシピ」**
  - **ViT-22B（Google, 2023）への対抗**: 21.7B パラメータ + 非公開 JFT-3B の ViT-22B に対し、**5.9B + 公開データ 4.98B** の InternViT-6B が ADE20K linear probe **47.2 vs 34.6（+12.6 mIoU）** という驚異的な差で勝利。**「公開データ + 適切なスケール + 適切な訓練 = JFT 不要」** を実証
  - **QLLaMA = 「QFormer の 42× 巨大化版」の意義**: BLIP-2 が 188M QFormer で「軽量 glue で十分」と主張したのに対し、InternVL は **「glue 自体を 8B スケールにして LLaMA 初期化すれば、LLM 接続も対比も生成もすべて 1 モジュールでこなせる」** と反論。CV における「glue layer scaling」議論の起点
  - **既存 wiki ハブとの自然な統合**: WSL 系（[[entities/clip]] / [[entities/siglip]] / [[entities/perception-encoder]]）の競合系列に位置づけ。**[[entities/perception-encoder]]（2025 NeurIPS）と並ぶ「6B+ 視覚エンコーダの対比学習スケール」軍備拡張** の先駆け。**[[entities/siglip|SigLIP 2]]（2025）** が「対比学習自体を改良（sigmoid + 全部入り）」なのに対し、InternVL は「対比 + 生成 + LLM 統合」軸。**[[entities/dinov2]] / [[entities/dinov3]]** との対比は「テキスト誘導 vs テキストなし純粋 SSL」の代表的競合軸
  - **動画 + 多言語のゼロショット強さ**: Kinetics-700 8F 60.6（ViCLIP +6.3）/ 多言語 IN-1K 平均 64.0（OpenCLIP-XLM-R-H +8.1）。**動画と多言語の両方で当時 SoTA** という稀有な複合性能
  - **OpenGVLab のシリーズ戦略**: InternImage（純粋画像）/ InternVideo（動画）/ InternVL（マルチモーダル）/ InternLM（言語）のシリーズが Shanghai AI Lab の Intern ブランドとして展開。**中国 AI 大規模研究機関の代表的成果**
  - 後続候補: InternVL 1.5（dynamic high-res, 2024-04）/ InternVL 2.0 / 2.5 / 3 series / Qwen-VL / Qwen2-VL / Gemini-Vision API 周辺 / LLaVA-NeXT / LLaVA-OneVision / MiniGPT-4 v2 / Flamingo / BLIP-2 原典 / EVA-CLIP / EVA-02 / MetaCLIP など

## [2026-05-29] ingest | InternVL 1.5: How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites

- 取り込み: `raw/papers/How Far Are We to GPT-4V_ Closing the Gap to Commercial Multimodal Models with Open-Source Suites.md`（ar5iv 由来 markdown、本文のみ、ユーザー指示で References は除外）
- 画像対応: ar5iv URL（<https://ar5iv.labs.arxiv.org/html/2404.16821/assets/x1-x5.png>）から 5 枚を curl でダウンロードし `raw/assets/internvl-1-5/fig{1-5}.png` に保存
- 作成:
  - [[translations/internvl-1-5]] — Abstract + §1-5 の全文和訳（References のみ除外）。図 1-5 を `<figure>` で埋め込み、表 1-3 を含む（表 2 は 18 ベンチ × 全商用・OS モデルの巨大マトリクス）。動的高解像度（35 種アスペクト比 × 1-40 タイル）、Pixel Shuffle、継続事前学習（InternViT V1.0 → V1.2 → V1.5）、QLLaMA 廃止 → MLP 統一、バイリンガルデータパイプライン、2 段階訓練、定性比較（vs GPT-4V）を含む
  - [[sources/internvl-1-5]] — 初学者向け要約。「**26B オープンソース MLLM で GPT-4V との差を 8/18 ベンチマークで埋めた到達点**」の位置づけ、3 つの改善（継続事前学習 + 動的高解像度 + バイリンガル）、QLLaMA 廃止の意義、ChartQA/OCRBench/CCBench/MathVista での商用超え、ConvBench での GPT-4V 大敗、定性比較（vs GPT-4V）、用語集（MLLM 関連 70+ 項目を網羅）
  - [[entities/internvl-1-5]] — InternViT-6B-448px-V1.5 / V1.2 / V1.0 比較、Pixel Shuffle 詳細、動的解像度 35 種、訓練 2 段階、全 18 ベンチ結果、vs InternVL 1.0 / vs LLaVA-NeXT-35B / vs GPT-4V の 3 アーキテクチャ比較、モデルカタログ（HuggingFace 公開リソース）、InternVL シリーズ全体（V1.0 → V3）の系譜
- 更新:
  - [[entities/internvl]] — InternVL 1.5 を「**InternVL シリーズ第 4 世代、CVPR 2024 とは別の technical report**」として系譜表に明示、QLLaMA 廃止の経緯を反映
  - [[sources/internvl]] — 「QLLaMA は InternVL 1.5 で完全廃止され LLaVA 系の MLP プロジェクタに統一された（軽量 glue で十分という業界トレンドの追認）」を明示、後続バージョン参照を更新
  - [[concepts/foundation-model]] — §A WSL 系セクションに InternVL 1.5 を追加
  - [[concepts/weakly-supervised-pretraining]] — §A CLIP 系セクションに InternVL 1.5 を追加（「対比学習基盤モデルから MLLM への完全シフトの転換点」）
  - [[concepts/zero-shot-transfer]] — InternVL 1.5 エントリ追加（「training 1-12 タイル → test 40 タイル zero-shot、single-image 訓練のみで multi-image zero-shot」）+ 関連ページに entities/internvl-1-5 追加
  - [[index]] — 3 つの新規ページ + 略称表に **50+ 項目**（InternVL 1.5 / 1.2 / InternViT V1.2/V1.5 / InternLM2-20B-Chat / Pixel Shuffle / Dynamic High-Resolution / 35 種アスペクト比 / Thumbnail / Continuous Learning / ViT-MLP-LLM / MLP プロジェクタ / GPT-4V / Gemini / Claude-3 / Qwen-VL-Max/Plus / Grok-1.5V / MM1 / Step-1V / HPT Pro / LLaVA-NeXT / Mini-Gemini / DocOwl-1.5 / Text-Monkey / CogVLM / DeepSeek-VL / LLaVA-HR / mixture-of-features / UReader / PaddleOCR / DocVQA / ChartQA / InfoVQA / OCRBench / MMBench / CCBench / MMMU / AI2D / MathVista / MMVet / SEED / RealWorldQA / HallusionBench / ConvBench / MMT-Bench / VLMEvalKit / OpenCompass / Wukong-OCR / LaionCOCO-OCR / GRIT / All-Seeing / SynthDoG / ALLaVA / ShareGPT4V / LVIS-Instruct4V / OpenHermes2.5 / COIG-CQIA / Nous-Hermes-2-Yi-34B）を新規登録
  - [[overview]] — VLM セクションに InternVL 1.5 を追加（InternVL 1.0 → InternVL 1.5 の進化を明示、GPT-4V との差を初めて縮めた MLLM として位置づけ）、ingest 済みリストを更新
  - [[log]]
- メモ:
  - **「QLLaMA 廃止」の戦略的意義**: InternVL 1.0（CVPR 2024）の「8B QLLaMA で対比 + 生成 + LLM 接続を統合」という野心的設計を、InternVL チーム自身が **1.2 で廃止し LLaVA 系の MLP プロジェクタに統一**。理由: (a) 訓練コスト高、(b) LLM 交換のたび QLLaMA 再訓練必要、(c) MLLM タスクでは MLP で十分。**業界トレンド（軽量 glue layer）への完全追随** であり、CV 系論文として希少な「自分の主要貢献を翌作で捨てる」事例
  - **「How Far Are We to GPT-4V?」の問いへの答え**: 「OCR と中国語ではもう追いついた、数学でも GPT-4V を上回った、しかしマルチターン対話と多分野知識（MMMU）ではまだ遠い」。**18 ベンチ中 8 SoTA は当時オープンソース MLLM の到達最高水準**
  - **動的高解像度の発明的価値**: 35 種アスペクト比 × 1-12 タイル（テスト 40 タイル）+ Pixel Shuffle で visual token 1/4 圧縮 = **「LLM 文脈長制約下で 4K 文書をどう扱うか」の標準解法を確立**。後の Qwen-VL2 / NVLM 等の dynamic resolution はこれに追随
  - **継続事前学習の威力**: InternViT-6B が V1.0（224、48 層、対比学習）→ V1.2（448、**45 層 = 48 - 3**、MLLM）→ V1.5（dynamic 448、45 層、bilingual）と進化。**「後ろから 4 層目が MLLM タスクで最良」** という発見は、[[entities/perception-encoder|PE]] の「最良特徴は中間層に育つ」発見の **約 1 年早い先取り**。CV foundation model における「中間層特徴量の優位性」の重要な実例
  - **バイリンガル OCR の実装の妙**: Wukong（華為の中国語 Web 画像）と LaionCOCO（英語 Web 画像）に **PaddleOCR で擬似 OCR ラベリング** する大胆な手法。さらに英語データセット（COYO, GRIT）を **LLM ベース翻訳パイプライン**（InternLM2 / Qwen / Yi-34B / GPT-3.5）で中国語化。**中国語 MLLM の OCR 性能を一気に商用超えに引き上げた** 鍵
  - **「Larger LLMs need Larger VFMs」の再確認**: アブレーション §4.3 で「6B VFM + 34B LLM vs 300M VFM + 34B LLM」を比較し、11 ベンチ中 9 で 6B VFM が勝利。**LLaVA 系が長年無視してきた「VFM スケーリングも重要」** という [[entities/internvl|InternVL 1.0]] の主張を再確認
  - **動的解像度はタスク依存**: OCR は高解像度 → 良くなるが、AI2D / MMMU / MMBench / HallusionBench は **高解像度でわずかに悪化**。「シーン推論にはグローバル context が大事、高解像度はそれを薄める」という直感的説明。**「すべてのタスクに高解像度を強要するべきでない」** という、後の MLLM 設計者への重要な教訓
  - **MMMU と MMT-Bench での退行は LLM サイズ縮小に起因**: InternVL 1.2 (40B, LLM=34B) → InternVL 1.5 (26B, LLM=20B) で MMMU -6.4 / MMT -4.4。**「多分野知識タスクでは LLM パラメータが効く」** という当然の事実の確認。InternVL 1.5 は 26B 軽量化と OCR・中国語特化を取って MMMU を捨てた戦略的判断
  - **ConvBench マルチターン対話で GPT-4V との差は大**: 17.65 vs 39.51 = **-21.86**。オープンソース中で首位だが、商用フロンティアとの差は依然として大きい。**single-image 訓練のみで multi-image にゼロショット対応するが、真の多ターン対話はまだ限界**
  - **既存 wiki ハブとの自然な統合**: [[entities/internvl]]（InternVL 1.0、CVPR 2024）の直接の発展形として位置づけ。**MLLM 系列（LLaVA-NeXT / Mini-Gemini / DocOwl-1.5 / DeepSeek-VL / CogVLM 等）** の wiki 化が初めて本格化、今後の MLLM ingest（Qwen-VL2, NVLM, LLaVA-OneVision 等）のハブが用意された
  - **OpenGVLab InternVL シリーズの全貌**: 1.0（2023-12, CVPR 2024）→ 1.2 + Plus（2024-02）→ **1.5（2024-04, 本論文）** → 2.0（2024-07）→ 2.5（2024-12）→ 3（2025-04）。**約 1 年半で 6 つのメジャーアップデート**、Shanghai AI Lab + SenseTime + 清華大学 + 香港中文大学の中国コアコンソーシアム
  - 後続候補: InternVL 2.0 / 2.5 / 3（OpenGVLab シリーズ続き）/ Qwen-VL / Qwen2-VL / Qwen2.5-VL（Alibaba 系）/ LLaVA-NeXT / LLaVA-OneVision / LLaVA-1.6（LLaVA 系）/ DeepSeek-VL / DeepSeek-VL2 / NVLM / Mini-Gemini / CogVLM / CogVLM2 / Phi-3-vision / Florence-2（小型）/ GPT-4V / Claude-3.5 Sonnet（商用、公開情報のみ）など

## [2026-05-29] ingest | Mini-InternVL: A Flexible-Transfer Pocket Multimodal Model with 5% Parameters and 90% Performance

- 取り込み: `raw/papers/Mini-InternVL_ A Flexible-Transfer Pocket Multimodal Model with 5% Parameters and 90% Performance.md`（ar5iv 由来 markdown、本文のみ、ユーザー指示で References は除外）
- 画像対応: ar5iv URL（<https://ar5iv.labs.arxiv.org/html/2410.16261/assets/x{1,2,3,4,6}.png>）から 5 枚を curl でダウンロードし `raw/assets/mini-internvl/fig{1-4,6}.png` に保存（fig5 は欠番）
- 作成:
  - [[translations/mini-internvl]] — Abstract + §1-5 の全文和訳（References のみ除外）。図 1-4 + fig6 を `<figure>` で埋め込み、表 1-13 を含む。InternViT-300M の知識蒸留（negative cosine similarity 損失、最後 K 層、CLIP-ViT-L 初期化）、Mini-InternVL の 2 段階訓練、統一ドメイン適応フレームワーク（5 種タスクの VQA 統一）、3 ドメイン（自律走行 / 医療 / リモセン）への転移結果、4 種類のアブレーション（KD 有無 / データ比率 / サンプル数 / 適応手法）を含む
  - [[sources/mini-internvl]] — 初学者向け要約。「**5% パラメータ × 90% 性能の軽量 MLLM スイート + 統一ドメイン適応フレームワーク**」の位置づけ、InternVL シリーズ二分岐の意義、3 ドメイン適応の詳細結果、アブレーション（KD 効果、データ比率、訓練手法）、用語集（MLLM/ドメイン関連 90+ 項目）
  - [[entities/mini-internvl]] — Mini-InternVL の 3 サイズ + DA 版モデルカタログ、訓練 2 段階 + ドメイン適応の Stage 3、統一フレームワーク 5 タスクの VQA 形式、全結果テーブル、vs InternVL 1.5 / vs Qwen2-VL-2B / vs MiniCPM-V 2.0 の比較
  - [[entities/internvit-300m]] — InternViT-300M 単独ページ。CLIP-ViT-L 初期化 + InternViT-6B 蒸留の詳細、蒸留損失（negative cosine similarity、最後 K 層）、訓練データ（5 カテゴリ）、vs CLIP-ViT-L / vs InternViT-6B 比較、系譜上の位置
- 更新:
  - [[entities/internvl]] — InternVL シリーズ表に Mini-InternVL を追加、**「InternVL の二分岐」**（InternVL 1.5 vs Mini-InternVL）を明示
  - [[entities/internvl-1-5]] — InternVL シリーズ系譜に Mini-InternVL を追加
  - [[concepts/knowledge-distillation]] — §1 モデル圧縮の章で InternViT-300M を具体例として明示（DINO-X Edge と並列）
  - [[concepts/foundation-model]] — §A WSL 系に Mini-InternVL を追加（「軽量化 + ドメイン特化」分枝として）
  - [[concepts/weakly-supervised-pretraining]] — §A CLIP 系に Mini-InternVL を追加（「VFM の知識蒸留で WSL の軽量化問題を解く」方向性）
  - [[index]] — 4 つの新規ページ（sources / translations / entities/mini-internvl / entities/internvit-300m）+ 略称表に **50+ 項目**（Mini-InternVL / DA バリアント / InternViT-300M / Pixel Unshuffle / Qwen2-0.5B / InternLM2-1.8B / Phi-3-Mini / DriveLM / DriveGPT4 / BDD-X / MME-RW / CAM 系 / GMAI-MMBench / PMC-OA/VQA/Image / MedICaT / MIMIC-CXR / Quilt-1M / LLaVA-Med / RadFM / GeoChat / RSVQA / FIT-RS / DIOR-RSVG / SkyEyeGPT / SkySenseGPT / Cambrian-1 / Qwen2-VL / LLaVA-OneVision / MiniCPM-V / DeepSeek-VL / Fuyu/MoMa/Chameleon / ZeRO1 / Negative Cosine Similarity Loss / Domain Adaptation 等）を新規登録
  - [[overview]] — VLM/MLLM セクションに Mini-InternVL を追加、Foundation Model 3 大系統に追加、ingest 済みリストを更新
  - [[log]]
- メモ:
  - **「InternVL シリーズの二分岐」**: InternVL 1.0 → 1.5 が「**商用追従 + 大規模化**」の方向に進化したのに対し、Mini-InternVL は **「軽量化 + ドメイン特化」** という別の方向に分岐。**InternViT-6B が教師として両系統の起点に位置する**ことが wiki 内で明示化。Shanghai AI Lab が InternVL シリーズで「両極をカバー」する戦略を採ったことが示される
  - **知識蒸留の現代的応用**: Mini-InternVL の最大の独自貢献は **「VFM の知識蒸留」**。CLIP-ViT-L-336px で初期化した 300M の生徒を、InternViT-6B（5.9B）から **最後の K 層 + negative cosine similarity** で蒸留。**「CLIP の初期表現 + InternViT-6B の多ドメイン知識」** という 2 段階で、CLIP 単体より大幅に強い軽量 VFM を構築。CLIP-ViT-L 単体との比較で OCR +8.4 / Chart +5.3 / InfoVQA +8.1 と顕著な優位。これは **CV foundation model の軽量化路線（MobileCLIP, CLIPA-v2 等）への重要な貢献**
  - **「5% パラメータで 90% 性能」のキャッチコピー**: Mini-InternVL-4B (4B) が InternVL2-Llama3-76B (76B) の **平均ベンチスコア 81.4 → 72.8（90%）** を達成、Gemini-Pro-1.5 (73.6) とほぼ同等。これは消費者 GPU（RTX 3090, 4090）での MLLM 展開可能性を初めて現実化した重要な結果
  - **統一ドメイン適応フレームワークの革新性**: 5 種類のタスク（画像分類 / grounding / 領域知覚 / 多視点 / 動画）を **VQA 形式** に統一する設計。**GeoChat / LLaVA-Med / DriveGPT4 が各々独自のフォーマットを採用** する中、Mini-InternVL は **「同一モデルアーキテクチャ + 同一データ形式 + 同一訓練スケジュール」** で複数ドメインに転移。これは **MLLM の "OS"** を作る試みであり、後の研究（MLLM の universal adapter 探求）に影響
  - **DriveLM での衝撃**: Mini-InternVL-DA-2B (**2B**) が CVPR 2024 Autonomous Driving Challenge SOTA の InternVL4Drive-v2 (**26B**) と **同等スコア 0.5958 vs 0.6002 を達成**。**1/13 のサイズで匹敵**。さらに MME-RealWorld 自律走行で DA-4B が **GPT-4o を +24.78 ポイント圧倒**（49.38 vs 24.60）、Claude 3.5 Sonnet も +17.28 圧倒。**「専門 MLLM より、ドメイン適応された汎用 MLLM の方が強い」** という重要な発見
  - **医療画像での商用超え**: Mini-InternVL-DA-4B が GMAI-MMBench の **2D Seg C / Seg M / Cls** で LLaVA-Med、RadFM (14B)、Claude3-Opus を凌駕。GPT-4V には届かないが、**4B の DA モデルが医療特化 14B モデルを超える** ことを実証。InternViT-300M に **PMC-VQA 等の医療データが蒸留段階で含まれている** ことが効いている
  - **リモートセンシングでの動的解像度の威力**: GeoChat / SkySenseGPT 等の既存リモセン MLLM は **単一解像度しか扱えない**。Mini-InternVL は **InternVL 1.5 由来の動的解像度** で高解像度衛星画像を効果的に処理、DIOR-RSVG で **92.04** を達成（SkyEyeGPT 88.59 超え）
  - **アブレーション結果の実用的価値**: (a) 知識蒸留が CLIP より OCR/Document/Chart で +8 ポイント前後の優位、(b) 汎用:特化 = 1:4（r=0.25）が自律走行で最適、(c) 訓練データの 1/4 で性能維持（計算 4× 削減）、(d) Full-parameter > Freezing ViT > LoRA の順、LoRA は grounding で弱い。これらは **後の MLLM ドメイン適応研究の標準的セッティング** になる可能性
  - **OpenGVLab の戦略**: InternViT-6B を **「教師として系列を統合する VFM」** として活用。InternVL 1.0 では対比 + 生成、InternVL 1.5 では継続事前学習、Mini-InternVL では蒸留。**1 つの強力な VFM が様々な派生形を生む** という戦略は、Meta の DINOv2/v3 や Google の SigLIP/SigLIP 2 とは異なるアプローチ
  - **LLM ベンダー選択の実用性**: Mini-InternVL は LLM 部分を **Qwen2-0.5B (Alibaba) / InternLM2-1.8B (Shanghai AI Lab) / Phi-3-Mini (Microsoft)** と異なるベンダーから採用。これは **「視覚側は内製で蒸留、LLM 側は外部から最良のものを選ぶ」** という分業を実証。Phi-3-Mini が「教科書データ訓練の強い軽量 LLM」だったことが Mini-InternVL-4B の MMMU 高スコアに効いた
  - **既存 wiki ハブとの自然な統合**: [[entities/internvl-1-5|InternVL 1.5]]（2024 April）の補完として位置づけ。「InternVL の二分岐」が wiki で明示化。**[[entities/internvit-300m|InternViT-300M]] という独立 entity ページ** を作ることで、CLIP / DINOv2 / SigLIP 等の VFM 比較群に **「KD で作られた軽量 VFM」** という新カテゴリを追加
  - 後続候補: InternVL 2.0 / 2.5 / 3 (OpenGVLab 続き) / Qwen2-VL 原典 / Qwen2.5-VL / DeepSeek-VL2 / NVLM / Phi-3-Vision 原典 / MiniCPM-V 2.6 / LLaVA-OneVision / Cambrian-1 / Apple AIMv2 / Apple MobileCLIP / Apple FastVLM など軽量 MLLM 系

## [2026-05-29] ingest | InternVL 2.5: Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling

- 取り込み: `raw/papers/Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling.md`（ar5iv 由来 markdown、本文のみ、ユーザー指示で References は除外）
- 画像対応: ar5iv URL（<https://ar5iv.labs.arxiv.org/html/2412.05271/assets/x1-x10.png>）から 10 枚を curl でダウンロードし `raw/assets/internvl-2-5/fig{1-10}.png` に保存
- 作成:
  - [[translations/internvl-2-5]] — Abstract + §1-8 + Acknowledgement の全文和訳（References のみ除外）。図 1-10 を `<figure>` で埋め込み、表 1-15 を含む（OpenCompass 順位、InternViT 全バージョン比較、モデルカタログ、訓練ハイパラ、Pre-train/Fine-tune データ、9 ベンチマーク群（Multimodal Reasoning / OCR / Multi-Image / Real-World / Comprehensive / Hallucination / Grounding / Multilingual / Video）、LLM 17 ベンチ、ImageNet 派生 6、ADE20K + COCO-Stuff）
  - [[sources/internvl-2-5]] — 初学者向け要約。「**MMMU で 70% を超えた初のオープンソース MLLM + 段階的スケーリング戦略を公式化**」の位置づけ、InternVL シリーズ商用追従ライン最終到達点、Progressive Scaling Strategy / Test-Time Scaling / データフィルタリング / InternViT 中間層特徴発見の詳細、用語集（MLLM/データ/ベンチ 100+ 項目）
  - [[entities/internvl-2-5]] — InternVL 2.5 の 7 サイズスイート + Pro 詳細仕様、InternViT 系統の系譜（V1.0-V2.5）、3 段階訓練 + Progressive Scaling Strategy、全結果（MMMU/MathVista/OCR/Grounding/Video/言語/視覚すべて）、vs InternVL 1.5 / 2.0 / GPT-4o / Claude-3.5-Sonnet の 4 比較、InternVL シリーズ系譜全体
- 更新:
  - [[entities/internvl]] — InternVL シリーズ表に InternVL 2.5 を追加、**「InternVL の二分岐を 2.5 が再統合（軽量も大規模も全部入り）」** を明示
  - [[entities/internvl-1-5]] — シリーズ系譜表に InternVL 2.5（MMMU 70.1%、Progressive Scaling）を追加
  - [[entities/mini-internvl]] — シリーズ系譜に InternVL 2.5（Mini-InternVL の軽量化路線を吸収）を追加
  - [[entities/internvit-300m]] — V2.5 への進化を追加（多様データで段階的事前学習）、系譜図に InternVL 2.5 を追加
  - [[concepts/foundation-model]] — §A WSL 系セクションに InternVL 2.5 を追加（MMMU 70% 突破、Progressive Scaling、Qwen2-VL 1/12 訓練トークン、PE と同等の中間層特徴発見）
  - [[concepts/weakly-supervised-pretraining]] — §A CLIP 系に InternVL 2.5 を追加（同じ哲学）
  - [[concepts/zero-shot-transfer]] — InternVL 2.5 エントリ追加（Test-Time Scaling: CoT + Majority Voting）+ 関連ページに entities/internvl-2-5 追加
  - [[concepts/alignment-tuning]] — 既存研究のヒントセクションに **「InternVL 1.5 の InternViT V1.2 → V1.5 の 48 → 45 層削除」** と **「InternVL 2.5 のセグ/分類アブレーション」** を **PE の中間層特徴発見の同時期独立観察** として追加
  - [[index]] — 3 つの新規ページ（sources/internvl-2-5、translations/internvl-2-5、entities/internvl-2-5）+ 略称表に **70+ 項目**（InternVL 2.5/2.0 / 7 サイズ / Pro / InternViT-V2.5 系 / InternLM 2.5 / Qwen 2.5 / NTP loss / QK-Norm / Pixel Unshuffle / Stage 1/1.5/2 / Progressive Scaling / Test-Time Scaling / CoT / Majority Voting / Random JPEG / Square Averaging / Data Packing / Repetition Detection / LLM Quality Scoring / ChatML / MMMU-Pro / MATH-Vision / MathVerse / OlympiadBench / CharXiv / VCR / SEED-2-Plus / Mantis-Eval / MMIU / MuirBench / BLINK / MIRB / WildVision / R-Bench / MMBench v1.1 / MMVet v2 / MMHal / CRPE / MMMB / Multilingual MMBench / MTVQA / Video-MME / MVBench / MMBench-Video / MLVU / LongVideoBench / CG-Bench / MMLU / CMMLU / C-Eval / GAOKAO / TriviaQA / NaturalQuestions / C3 / RACE / WinoGrande / HellaSwag / BBH / GSM8K / MATH / TheoremQA / HumanEval / MBPP / Linear Probing / Attention Pooling Probing / Head Tuning / Full Tuning / UperNet / Grounding-DINO-L / UNINEXT-H / ONE-PEACE / TextHawk2 / Ferret-v2 / CogVLM-Grounding / NVLM / Molmo / LLaVA-OneVision / MiniCPM-V 2.6 / Ovis 1.6 / Phi-3.5-Vision / Aquila-VL / Cambrian / VILA-1.5 / InternVL4Drive-v2 / OpenAI o1）を新規登録
  - [[overview]] — VLM/MLLM セクションに InternVL 2.5 を追加（InternVL 1.5 → Mini-InternVL → InternVL 2.5 の進化を明示）、Foundation Model 3 大系統に追加、ingest 済みリスト更新
  - [[log]]
- メモ:
  - **「MMMU 70% 突破の象徴的意義」**: 2024 年中盤まで MMMU は「商用 vs オープン」の最後の砦の 1 つで、GPT-4o (69.1) / Claude-3.5-Sonnet (68.3) / Gemini-1.5-Pro (62.2) というラインがあった。InternVL2.5-78B が **70.1** で **初めてオープン MLLM が 70 を超え**、商用フロンティアとの差を完全に閉じた瞬間。MLLM 競争の象徴的な転換点
  - **「アーキテクチャは変えず、訓練・データ・テスト時で勝つ」哲学**: InternVL 2.5 は [[entities/internvl-1-5|InternVL 1.5]] と完全に同じ ViT-MLP-LLM 構造を保持。**「モデル構造の革新ではなく、データ品質 + 段階的訓練 + テスト時推論で性能境界を拡張」** という哲学。これは [[sources/internvl-1-5|InternVL 1.5]] が「QLLaMA 廃止 → MLP に統一」と構造を簡素化した方向性の延長
  - **Progressive Scaling Strategy の公式化**: 小型 LLM（20B）で InternViT を訓練し、その後大型 LLM（72B）に転送する戦略。**ViT と LLM の共同訓練が「他の LLM が容易理解できる汎用視覚特徴」を生む** という観察に基づく。**Qwen2-VL の 1/12 の訓練トークン**（120B vs 1.4T）で同等以上の性能。これは [[sources/internvl-1-5|InternVL 1.5]] と Mini-InternVL の経験を一般化した方法論
  - **Test-Time Scaling の MLLM 適用**: 2024 年 9 月の OpenAI o1 以来、LLM の test-time scaling は中心話題だった。本論文は MLLM への適用を本格的に検証した初期論文の 1 つ。**CoT で MMMU +3.7、Majority Voting でさらに改善**。「OS MLLM も test-time scaling の恩恵を受けられる」ことの実証
  - **データフィルタリングの教訓**: **「LLM は視覚エンコーダよりノイズに敏感、数千の繰り返しパターンサンプルでも CoT 推論が無限ループに陥る」** という発見が衝撃的。InternVL 2.0 は CoT 不安定だったが、2.5 は厳格フィルタリング（LLM スコアリング + 繰り返し検出 + ヒューリスティック規則）で解消。**「データ品質 > データ量」** の MLLM 文脈での確認
  - **InternViT の「中間層特徴」発見の独立性**: 表 14 / 15 で **InternViT のバージョン更新（V1.0 → V2.5）で linear probing が継続的に低下、attention pooling / head tuning は維持または改善、Δ（差）が拡大** という現象を明確に観察。これは [[sources/perception-encoder|PE]]（NeurIPS 2025）の「対比学習をスケールすると中間層に多目的特徴が育つ」発見と本質的に同じ。**2024 年中盤に複数の独立チームが同じ現象を観察した** ということ。**[[concepts/alignment-tuning]] への追記で両者の同期発見を明示化**
  - **「6B 視覚エンコーダの威力」の再主張**: Qwen2-VL-72B（600M 視覚）vs InternVL2.5-78B（5.5B 視覚）の **訓練トークン 10×、性能同等以上** の結果は、[[entities/internvl|InternVL 1.0]] からの一貫主張「**Larger LLMs need Larger VFMs**」の最終的な実証。ただし 2B 級では Qwen2-VL-2B（600M 視覚 + 1.5B LLM）が OCR で優位、と例外も認める
  - **データセット規模の段階的増加**: InternVL 1.5（5.1M） → 2.0（7.3M） → **2.5（16.3M）**。**動画 +30% / 複数画像 +5%** が動画理解と複数画像理解の劇的改善に直結
  - **モデルファミリーの完全性**: 1B/2B/4B/8B/26B/38B/78B の **7 サイズスイート** で、消費者 GPU（1B-4B）からエンタープライズ（78B）までカバー。[[entities/mini-internvl|Mini-InternVL]] の軽量化路線を InternVL 2.5 が**正式に吸収**（Mini-InternVL は InternVL 2.0 と同期生成だった）
  - **VCR タスクの「22K サンプルで +60 ポイント」**: InternVL2-2B (32.9) → InternVL2.5-2B (93.2) という大幅改善は **「VCR の本質的能力欠如ではなく、指示追従能力不足」** だったという発見。Visual Caption Restoration（画像内の隠されたテキストを復元）は MLLM の OCR + 指示理解能力の精密診断ツール
  - **InternViT-300M-V2.5 の意義**: [[entities/mini-internvl|Mini-InternVL]] (2024 Oct) で蒸留された InternViT-300M を、InternVL 2.5 (2024 Dec) でさらに多様データで段階訓練して V2.5 へ。**「蒸留 + 継続事前学習」** の方法論が確立。これは Mini-InternVL と InternVL 2.5 が **同じ視覚エンコーダ系統で共生** する状態を作った
  - **言語能力の保持・強化**: InternVL 2.0 では基盤 LLM より純粋言語性能が -2.1〜-2.3 低下していたが、2.5 では **+0.5〜+1.4 改善**。MLLM 訓練で言語性能を保持できることを実証、これは「マルチモーダル化のコスト」議論への重要な反論
  - **限界の自覚**: 著者は **WildVision で GPT-4o に -9.2、MMVet v2 で -5.5** という長応答品質・統合能力の差を明記、Stage 3（DPO/RLHF/preference optimization）が未実施であることを認める。「**簡潔・正確な応答は OS が追いついた、長応答とユーザ嗜好整合は次の課題**」という現状把握
  - **既存 wiki ハブとの自然な統合**: [[entities/internvl]]（CVPR 2024）→ [[entities/internvl-1-5]]（2024-04）→ [[entities/mini-internvl]]（2024-10）→ [[entities/internvl-2-5]]（2024-12）と、InternVL シリーズの wiki が時系列に統合された。**6 世代の系譜 + 7 サイズ × 8 モデルカタログ** が wiki 内で追跡可能。次の InternVL 3（2025-04）への準備完了
  - 後続候補: InternVL 3 (Native multimodal pretraining、2025-04、InternVL シリーズ続き) / Qwen2-VL / Qwen2.5-VL / DeepSeek-VL2 / NVLM 原典 / Phi-3.5-Vision 原典 / MiniCPM-V 2.6 / LLaVA-OneVision 原典 / Cambrian-1 原典 / Aquila-VL / Molmo 原典 / OpenAI o1 関連（test-time scaling 系列）など

## [2026-05-29] ingest | InternVL 3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models

- 取り込み: `raw/papers/InternVL3_ Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models.md`（ar5iv 由来 markdown、本文のみ、ユーザー指示で References は除外）
- 画像対応: ar5iv URL（<https://ar5iv.labs.arxiv.org/html/2504.10479/assets/x1-x3.png>）から 3 枚を curl でダウンロードし `raw/assets/internvl-3/fig{1-3}.png` に保存
- 作成:
  - [[translations/internvl-3]] — Abstract + §1-4 + Conclusion の全文和訳（References のみ除外）。図 1-3 を `<figure>` で埋め込み、表 1-13 を含む（7 サイズモデルカタログ、Multimodal Reasoning + Math + VisualPRM-Bo8、OCR/Chart/Document、Multi-Image + Real-World、Comprehensive + Hallucination、Grounding、Multilingual、Video、GUI Grounding、VSI-Bench 空間推論、LLM 17 ベンチ、V2PE δ アブレーション、MPO アブレーション）
  - [[sources/internvl-3]] — 初学者向け要約。「**Native Multimodal Pre-Training パラダイムを確立、MMMU 72.2 で SOTA 更新**」の位置づけ、InternVL シリーズの哲学転換（事後的 MLLM 適応 → Native）、V2PE / MPO / VisualPRM の詳細、Qwen2.5-Chat より強い言語能力の発見、用語集（80+ 項目）
  - [[entities/internvl-3]] — InternVL 3 の 7 サイズ仕様、3 段階訓練（Native Pre-Training + SFT + MPO）、V2PE 数式、VisualPRM-8B + Best-of-N、全 12 ベンチマーク結果、vs InternVL 2.5 / Qwen2.5-VL-72B / 商用フロンティア（GPT-4o/Claude-3.7-Sonnet/Gemini-2.0-Pro）の 3 比較、InternVL シリーズ系譜全体
- 更新:
  - [[entities/internvl]] — InternVL シリーズ表に InternVL 3 を追加、**「InternVL の二分岐 → 再統合 → パラダイム転換」** を明示（「InternVL 2.5 で軽量と大規模を再統合、InternVL 3 で事後的 → Native 哲学転換」）
  - [[entities/internvl-1-5]] — シリーズ系譜表に InternVL 3 を追加（Native Multimodal Pre-Training）
  - [[entities/mini-internvl]] — シリーズ系譜に InternVL 3 を追加
  - [[entities/internvl-2-5]] — シリーズ系譜に InternVL 3 を追加、「InternVL 2.5 は商用追従ラインの最終到達点、次の 3 で哲学転換」を明示
  - [[entities/internvit-300m]] — InternVL 3 でも V2.5 を継承を追加、「2 世代にわたって使われる成熟版」を明示
  - [[concepts/foundation-model]] — §A WSL 系セクションに InternVL 3 を追加（Native Multimodal Pre-Training、V2PE、MPO、VisualPRM、「マルチモーダル化で言語が強くなる」初実証）
  - [[concepts/weakly-supervised-pretraining]] — §A CLIP 系に InternVL 3 を追加（WSL 文脈での「言語能力強化」初実証を強調）
  - [[concepts/zero-shot-transfer]] — InternVL 3 エントリ追加（VisualPRM + Best-of-N、OpenAI o1 / DeepSeek R1 系の test-time scaling を MLLM へ本格適用）+ 関連ページに entities/internvl-3 追加
  - [[index]] — 3 つの新規ページ（sources/internvl-3、translations/internvl-3、entities/internvl-3）+ 略称表に **45+ 項目**（InternVL 3 / 7 サイズ / InternLM3-8B / Qwen2.5 base / Native Multimodal Pre-Training / V2PE / δ / MPO / DPO / BCO / VisualPRM / VisualPRM-8B/400K / MMPR v1.2 / Best-of-N / InternEVO / Head-Parallel / InternVL-Data / GUI Grounding / ScreenSpot / UI-TARS-72B / Aguvis-72B / VSI-Bench / MathVision / DynaMath / WeMath / LogicVista / Claude-3.7-Sonnet / Gemini-2.0-Pro / Gemini-2.5-Pro / Qwen2.5-VL 系 / QvQ-72B / Ovis2 / MiniCPM-o2.6 / Oryx-1.5 / VideoLLaMA2 / OmniCorpus / Tongyi Qianwen LICENSE / GLM-4v-Plus / Step-1o / ChatGPT-4o-latest）を新規登録
  - [[overview]] — VLM/MLLM セクションに InternVL 3 を追加、Foundation Model 3 大系統に追加、ingest 済みリスト更新
  - [[log]]
- メモ:
  - **「InternVL シリーズの哲学転換」**: InternVL 1.0 (CVPR 2024) から 2.5 (2024 Dec) まで、すべて **「LLM Chat 版から MLLM を事後改造」** という伝統的パラダイムだった。InternVL 3 は **「Qwen2.5 base + テキスト + マルチモーダル共同事前学習」** に転換、**「アーキテクチャは [[entities/internvl-1-5|InternVL 1.5]] と同じだが、訓練哲学は全く違う」** という極端な変化。CV 系列の論文として希少な「自分のシリーズの伝統を否定する」事例
  - **「マルチモーダル化で言語が強くなる」初実証**: 表 11 の結果が衝撃的。**Qwen2.5-Chat（言語特化）を、同じ Qwen2.5 base から派生した InternVL 3（マルチモーダル統合）が 17 ベンチの平均で +1.6 〜 +8.9 上回る**（小型ほど顕著）。これは **「マルチモーダル化は言語能力にとってコスト」という 5 年間の常識への重要な反論**。要因: 25% 純粋言語データの統合 + 共同パラメータ最適化 + 高品質テキスト後訓練データ
  - **V2PE の発明的価値**: 視覚トークンに **δ < 1 の位置インクリメント** を使うアイデア。**「視覚トークンは画像の隣接パッチ間に強い局所的相関があり、テキストトークンほど位置の独立性がない」** という洞察に基づく。$\delta = 1/4$ で最良、**短文脈タスクでも +0.7 改善**（表 12）。後続研究への重要な参照点
  - **MPO（Mixed Preference Optimization）の意義**: 単純な DPO（preference）+ BCO（quality）+ LM（generation）の組み合わせで **+4.5 ポイント改善**（38B モデル）。**MPO データは SFT データのサブセット** → 改善はアルゴリズムによる（データではない）ことを実証。OpenAI o1 / DeepSeek R1 系の reasoning 強化を MLLM へ本格的に適用
  - **VisualPRM-8B の独自貢献**: **Process Reward Model（PRM）を MLLM 向けに導入した初の本格事例**。各推論ステップに +/- スコアを付与し平均してソリューション全体スコアを算出。**Best-of-N で InternVL3-1B が +9.9 ポイント** という、**「計算リソースをモデルサイズで使うか、推論回数で使うか」** という新しいトレードオフ。OpenAI o1 / DeepSeek R1 系の test-time scaling を視覚に拡張
  - **MMMU 72.2 で SOTA 更新**: InternVL2.5-78B 70.0 → 72.2 (+2.2)、Qwen2.5-VL-72B 68.2 / GPT-4o 70.7 / Gemini-2.0-Pro 69.9 を超え、**Claude-3.7-Sonnet 75.0 のみ上**。MMMU は依然として「最後の砦」だが、オープン MLLM がさらに突破を続けている
  - **MathVista で GPT-4o を +19.0 圧倒**: MathVista 79.0（GPT-4o 60.0、Claude-3.7-Sonnet 66.8、Gemini-2.0-Pro 71.3 全部超え）。数学領域では完全に商用を超えた水準
  - **OCRBench 906 で史上初の 900 超え**: Qwen2.5-VL-72B 885、InternVL2.5-78B 854 を超え、史上初の 900 超え。OCR 領域の到達点
  - **VSI-Bench 空間推論で GPT-4o を +8.1 圧倒**: InternVL3-8B (42.1) > GPT-4o (34.0)、InternVL3-38B (48.9) > Gemini-1.5 Pro (45.4)。**3D シーン理解の新水準**、自律走行・ロボティクス応用への大きな前進
  - **訓練データ完全公開**: `OpenGVLab/InternVL-Data` で **InternVL3 の訓練データを完全公開**。"open-science" 原則を強調。これは Mini-InternVL / InternVL 2.5 でも完全には公開していなかった点を改善
  - **Visual Grounding でシリーズ初の退行**: InternVL2.5-78B 92.3 → InternVL3-78B 91.4 (-0.9)。著者自身が「**訓練データ拡張で grounding 特化データが含まれず、相対的に grounding データ比率が低下したため**」と分析。シリーズ初の「先代に劣る」ケース、将来課題
  - **LLM ベンダー戦略**: InternVL 2.5 では InternLM 2.5 系を主軸にしていたが、**InternVL 3 では Qwen2.5 base を主軸に切り替え**（7 モデル中 6 つが Qwen2.5、1 つのみ InternLM3-8B）。**Shanghai AI Lab が自社 LLM への過度な依存を緩和** する戦略的判断。これは Qwen2.5 が 2024 末で最強の OS LLM だったことを反映
  - **「base モデル + 共同事前学習」の意義**: Chat 版 LLM は既に「人間の好み」に調整されており、視覚情報を組み込む際に **「言語の硬直性」が干渉** する。Base モデルは「素の言語能力」のみで、視覚モダリティ統合の余地が大きい。これは画家が **真っ白なキャンバスから絵を描き始める** か、**既存の絵に追加描画する** かの違いに似ている
  - **既存 wiki ハブとの自然な統合**: InternVL シリーズの wiki が時系列に統合（[[entities/internvl]] → [[entities/internvl-1-5]] → [[entities/mini-internvl]] → [[entities/internvl-2-5]] → **[[entities/internvl-3]]**）。**7 世代の系譜 + 各世代でのモデルカタログ** が wiki 内で追跡可能。InternVL シリーズ ingest 完了
  - 後続候補: Qwen2.5-VL 原典（InternVL 3 の直接競合）/ Qwen2-VL 原典 / DeepSeek-VL2 / NVLM 原典 / Phi-3.5-Vision 原典 / MiniCPM-V 2.6 原典 / LLaVA-OneVision 原典 / Cambrian-1 原典 / Aquila-VL 原典 / Molmo 原典 / UI-TARS / Aguvis 原典（GUI agent 特化）/ OpenAI o1 原典 / DeepSeek R1 原典（test-time scaling 系列）/ VisualPRM 原典 / MMPR 原典 / MPO 関連論文 / V2PE 原典など

## [2026-05-29] ingest | MPO: Enhancing the Reasoning Ability of MLLMs via Mixed Preference Optimization

- 取り込み: `raw/papers/Enhancing the Reasoning Ability of Multimodal Large Language Models via Mixed Preference Optimization.md`（ar5iv 由来 markdown、本文 §1-6 + 実装詳細 §7 + 追加アブレーション §8 概要を ingest、References と §9 Appendix データ例図は除外）
- 画像対応: ar5iv URL（<https://ar5iv.labs.arxiv.org/html/2411.10442/assets/x{1,2,3,4}.png>）から 4 枚を curl でダウンロードし `raw/assets/mpo/fig{1-4}.png` に保存（§9 のデータ例 fig 5-17 は除外）
- 作成:
  - [[translations/mpo]] — Abstract + §1-6 + Implementation Details + More Ablation Studies 概要の全文和訳（References と §9 Appendix データ例図は除外）。図 1-3 を `<figure>` で埋め込み、表 1-9 を含む（PO アルゴリズム比較、SFT vs MPO、RLAIF-V vs DropoutNTP、データソース、テキスト専用ベンチ、X+ 拡張、DPO 変種比較、Dropout Ratio アブレーション、データスケール、ハイパラ）
  - [[sources/mpo]] — 初学者向け要約。「**MLLM の CoT 推論で性能が悪化する現象を初めて本格的に解決した論文**」の位置づけ、MPO アルゴリズム + MMPR + DropoutNTP の 3 点セット、SFT loss が CoT 強化の鍵、InternVL シリーズでの位置づけ、用語集（80+ 項目）
  - [[entities/mpo]] — MPO アルゴリズム単独ページ。3 損失の数式と役割、訓練設定、10 PO 手法のアブレーション比較、InternVL2-8B-MPO の全結果、InternVL シリーズでの位置（InternVL 3 Stage 3 で正式採用）
  - [[entities/mmpr]] — MMPR データセット単独ページ。3M サンプルの内訳、2 つの構築パイプライン（正誤判定 + DropoutNTP）、18 データソース × 6 ドメイン、Dropout Ratio アブレーション、RLAIF-V 比較、3 種マルチモーダル CoT、データスケーリング効果
- 更新:
  - [[entities/internvl-3]] — Stage 3 MPO 説明に **MPO 起源（本論文 = Wang et al., 2024 Nov）** を明示、[[entities/mpo]] と [[entities/mmpr]] へのリンク追加
  - [[concepts/foundation-model]] — §A WSL 系セクションに MPO 論文を追加（CoT 推論問題の解決、InternVL 3 への波及）
  - [[concepts/zero-shot-transfer]] — MPO エントリ追加（マルチモーダル推論能力強化の PO、InternVL 3 Stage 3 採用）
  - [[index]] — 3 つの新規ページ（sources/mpo、translations/mpo、entities/mpo、entities/mmpr）+ Datasets セクションに MMPR を追加 + 略称表に **40+ 項目**（MPO 論文 / MMPR v1 / v1.2 / DropoutNTP / InternVL2-8B-MPO / BCO / RLAIF-V / Teacher Forcing / Distribution Shift / Bradley-Terry / Reference Model / Policy Model / KL Penalty β / Reward Shift δ / RSO / IPO / cDPO / RobustDPO / SPPO / AOT / TR-DPO / ORPO / Smaug / Object HalBench / InstructGPT / PPO-Max / M3CoT / Geo170K / GeoQA+ / GEOS / GeomVerse / Geometry3K / CLEVR-Math / DVQA / MapQA / OCRVQA / STVQA / SROIE / TheoremQA / IFEval / GAOKAO / 3H / Background Knowledge-based CoT / Visual Content-based CoT / Grounded CoT / Dropout Ratio / token cost per pair）を新規登録
  - [[overview]] — VLM/MLLM セクションに MPO 論文を追加（InternVL 2 系列の CoT 弱点を解決、InternVL 3 で正式採用）、ingest 済みリスト更新
  - [[log]]
- メモ:
  - **「MLLM の CoT 推論が悪化する」謎の解決**: 図 1 が示すように、多くのオープン MLLM（LLaVA-1.5-13B / Qwen2-VL-7B / MiniCPM-V-2.6-8B / InternVL2-8B）が **CoT で性能低下**。LLM の世界では「CoT は推論を改善する」が常識だったため、MLLM の例外性が問題だった。著者らは **「SFT が teacher forcing による分布シフトを生み、長い CoT rationale で誤差蓄積する」** と分析。これは MLLM 訓練パイプラインの根本的な制約を初めて言語化した重要な観察
  - **MPO の中心的洞察**: 「効果的な PO は 3 つを学習すべき」 → **相対選好 + 絶対品質 + 生成過程**。これを **DPO + BCO + SFT loss** の混合で実現。**$w_p=0.8, w_q=0.2, w_g=1.0$** という重み配分は、preference を重視しつつ quality と generation で補完するバランス
  - **10 PO アルゴリズム比較の意義**: DPO / RSO / IPO / cDPO / RobustDPO / BCO / SPPO / AOT / TR-DPO / ORPO の **全 10 種を体系的に比較** したのは本論文が初。これは MLLM コミュニティへの重要な貢献。「DPO 単体は CoT 改善せず」「SFT loss が CoT 強化の鍵」「参照モデル凍結が必須」という 3 つの法則を発見
  - **DropoutNTP の天才的なアイデア**: 「画像入力なしで応答を補完すると、モデルは画像の中身を **推測（hallucination）** で補完する。その推測は元応答より**必ず悪い**」という構造的性質を利用。**人手アノテーション不要で大規模に rejected を生成** できる。**Dropout Ratio 0.5 が最適**（0.25 は品質差大きすぎ、0.75 は共通プレフィックスが多すぎ）
  - **RLAIF-V の 57.5% コスト**: 選好ペアあたり 571.2 トークン vs RLAIF-V 992.7 トークン、性能ほぼ同等（Object HalBench 7.6 vs 7.3、MMHal-Bench 3.6 vs 3.5）。**マルチモーダル選好データ構築の標準コストを大幅削減**
  - **MathVista +8.7 で 76B 同等の衝撃**: InternVL2-8B-MPO が MathVista **67.0**、InternVL2-76B の **67.2 とほぼ同等**。**10× のパラメータ差を MPO で埋めた** という、当時のオープン MLLM コミュニティに大きな衝撃を与えた結果
  - **M3CoT +19.9 という驚異**: 59.3 → 79.2。**マルチモーダル CoT ベンチで 20 ポイント近い改善**は前例がない
  - **テキスト専用ベンチでも改善**: MMPR にテキスト専用データなしにもかかわらず、**TheoremQA +5.2、IFEval +4.1、平均 +0.9**。これは [[entities/internvl-3|InternVL 3]] の「Native Multimodal Pre-Training で言語が強くなる」発見の **重要な先例**。「マルチモーダル選好最適化は言語能力にも波及する」というメカニズムを初観察
  - **InternVL 3 での MPO 正式採用**: 本論文の発表（2024 Nov）から 5 ヶ月後の InternVL 3（2025 Apr）で、**MPO が Stage 3 として正式採用**。InternVL3-78B で **+4.1 ポイント推論改善**、38B で +4.5。**MMPR v1.2（拡張版）+ 300K 選好ペア + VisualPRM400K** という形で発展。InternVL シリーズの永続的技術となった
  - **「事後的解決」vs「根本的解決」**: MPO は SFT 後の追加段階として実施 = 事後的解決。InternVL 3 の Native Multimodal Pre-Training は **「事前学習段階から共同最適化」** という根本的解決。両者は補完的（InternVL 3 でも Stage 3 として MPO を併用）
  - **MLLM 訓練パイプラインの第 3 段階確立**: LLaVA 系の MLLM は事前学習 + SFT の 2 段階で完結していたが、**LLM 界では DPO/RLHF が既に標準**。本論文は **その方法論を MLLM へ持ち込み、CoT 推論の問題を解決した先駆け**。後の MLLM 論文の多くが MPO 系統の post-training を組み込むようになる
  - **VisualPRM への進化**: 本論文の MMPR が **VisualPRM-8B**（[[entities/internvl-3|InternVL 3]] の Process Reward Model）の訓練データの基盤となる。VisualPRM400K は MMPR v1.2 から構築されており、**「選好データを critic モデル訓練にも活用」** という展開
  - **既存 wiki ハブとの自然な統合**: [[entities/internvl-3]] の Stage 3 MPO 説明と、[[entities/mpo]] / [[entities/mmpr]] の独立ページが相互リンク。**「MPO は InternVL 3 の一部分ではなく、独立した研究貢献」** という構造を wiki で表現
  - 後続候補: VisualPRM 原典 / DeepSeek R1 原典（PRM 系の MLLM 拡張）/ Process Reward Model 一般論 / RLAIF-V 原典 / Silkie / POVID / RLHF-V（マルチモーダル選好データ手法比較）/ DPO 原典（Rafailov et al., 2023）/ BCO 原典（Jung et al., 2024）/ PPO / InstructGPT 原典など RL/PO 関連の基礎論文

## [2026-05-29] ingest | InternVL 3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency

- 取り込み: `raw/papers/InternVL3.5_ Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency.md`（ar5iv 由来 markdown、本文のみ、ユーザー指示で References は除外）
- 画像対応: ar5iv URL（<https://ar5iv.labs.arxiv.org/html/2508.18265/assets/x{1-5}.png>）から 5 枚を curl でダウンロードし `raw/assets/internvl-3-5/fig{1-5}.png` に保存
- 作成:
  - [[translations/internvl-3-5]] — Abstract + §1-4 の全文和訳（References は除外）。図 1-5 を `<figure>` で埋め込み、表 1-18 を含む（9 サイズモデルカタログ、全 35 ベンチマーク比較、Multimodal Reasoning + Math、OCR/Chart/Document、Multi-Image + Real-World、Comprehensive + Hallucination、Grounding、Multilingual、Video、GUI、Embodied、SVG、LLM 16 ベンチ、Cascade RL アブレーション、Cascade RL vs MPO vs GSPO 効率、ViR Flash 性能維持、DvD + ViR 加速）
  - [[sources/internvl-3-5]] — 初学者向け要約。「**Cascade RL + ViR + DvD + MoE で 4.05× 加速、GPT-5 との差 3.9%**」の位置づけ、InternVL シリーズ第 8 世代、Cascade RL（MPO + GSPO）の coarse-to-fine 戦略、ViR/ViCO の数式、DvD のシステム最適化、MoE スケーリング、Qwen3 base 統一、用語集（100+ 項目）
  - [[entities/internvl-3-5]] — InternVL 3.5 の 9 サイズスイート + Flash 詳細仕様、Dense 6 + MoE 3 + Flash 効率版の全仕様、4 段階訓練（Native Pre-Training + SFT + Cascade RL + 任意で ViCO）、Cascade RL の MPO + GSPO 詳細、全 13 ベンチマーク結果、vs InternVL 3 / vs GPT-5 の 2 比較、InternVL シリーズ全 8 世代系譜
- 更新:
  - [[entities/internvl]] — InternVL シリーズ表に InternVL 3.5 を追加、**第 8 世代の位置づけ明示**
  - [[entities/internvl-3]] — シリーズ系譜表に InternVL 3.5（Cascade RL、MoE、ViR/DvD）を追加
  - [[entities/mpo]] — シリーズでの位置に InternVL 3.5 を追加（MPO が Cascade RL の Stage 1 として進化）
  - [[concepts/foundation-model]] — §A WSL 系セクションに InternVL 3.5 を追加（Cascade RL、MoE、ViR/DvD、4.05× 加速、GPT-5 との差 3.9%）
  - [[concepts/zero-shot-transfer]] — InternVL 3.5 エントリ追加（Deep Thinking + Parallel Thinking + Cascade RL）+ 関連ページに entities/internvl-3-5 追加
  - [[index]] — 3 つの新規ページ（sources/internvl-3-5、translations/internvl-3-5、entities/internvl-3-5）+ 略称表に **60+ 項目**（InternVL 3.5 / 9 サイズ / Flash / Cascade RL / GSPO / GRPO / Importance Ratio / ViR / ViCO / patch router / loss ratio / Pixel Shuffle 1/16 / DvD / Vision Server / Language Server / 非同期 3 段階パイプライン / RDMA / Qwen3 / GPT-OSS / MoE Activated Params / Thinking Mode / Deep Thinking / Parallel Thinking / VisualPRM-v1.1 / MMPR-Tiny / InternEVO / XTuner / FSDP / DeepGEMM / liger-kernel / FlashAttention-3 / TMA-Adaptive FP8 / verl / window attention with sink / GPT-5 / GLM-4.1V/4.5V / Step-3 / Kimi-VL / MiMo-VL-RL / Keye-VL / Ovis 系 / MiniCPM 系 / Skywork-R1V3 / QvQ / Doubao / Seed1.5-VL / WindowsAgentArena / WebArena-Lite-v2 / OSWorld / SGP-Bench / SArena-Icon / ERQA / SpaCE-10 / OmniSpatial / Llama-4 / DeepSeek-V3 / Reward Hacking / Request Throughput）を新規登録
  - [[overview]] — VLM/MLLM セクションに InternVL 3.5 を追加（InternVL シリーズ第 8 世代として明示）、Foundation Model 3 大系統に追加、ingest 済みリストを更新
  - [[log]]
- メモ:
  - **「商用 GPT-5 との差をオープンソース最小に縮める」**: GPT-5 (2025 Aug 7) の直後の発表（同月 26 日）で、**Aggregate 差 3.9%** という当時オープンソース最小を達成。InternVL3.5-241B-A28B が一般マルチモーダルで GPT-5 と互角（74.1 vs 74.0）、MathVista で +0.8 上回り、VSI-Bench で **+32 ポイント圧倒**。商用フロンティアとの最終決戦の様相
  - **Cascade RL の天才的なアイデア**: 「MPO は性能上限低いが安定・高速、GSPO は性能上限高いが不安定・遅い → 順番に適用」。MPO で warm-up したモデルは **高品質 rollouts を生成** → GSPO の online RL を **少ない episode で収束** させられる。これは **「コスト × 性能」の Pareto 改善**。GSPO 単独の半分の GPU 時間で +2.1 ポイント上回るという結果は印象的
  - **MoE スケーリングの本格採用**: InternVL シリーズで初の **MoE モデル 3 サイズ**（20B-A4B / 30B-A3B / **241B-A28B**）。activated parameter で実用効率を保ちつつ、合計 241B という巨大スケールを実現。**OpenAI の GPT-OSS-20B を InternVL3.5-20B-A4B に統合** したのも注目（OpenGVLab 系列で初の OpenAI 公開モデル統合）
  - **ViR + ViCO の意味的革新**: 従来の Dynamic High Resolution が **画像の幅と高さ（spatial）** で patch を分割していたのに対し、**ViR は意味的内容（semantic）** で動的に圧縮率を選択。「semantic-rich patch は 256 token、semantic-poor patch は 64 token」という直感的な発想を **loss ratio による learnable thresholding** で実装。**視覚トークン 50% 削減で性能 99% 維持** は実用展開に大きい
  - **DvD のシステムレベル最適化**: これは **MLLM のシステムレベル最適化** の本格的試みであり、**「アルゴリズムではなくインフラで性能を上げる」** という新方向。MoE 時代の大規模 MLLM 展開で必須技術になる可能性。Vision Server と Language Server を **BF16 視覚特徴を TCP/RDMA で転送** という設計は、production レベルでのデプロイを意識した精緻な実装
  - **「マルチモーダル化で言語能力が強くなる」の継続実証**: InternVL 3 で発見された現象が InternVL 3.5 でも継続。**Qwen3-base 比で 1B が +6.7、241B が +2.3、16 ベンチ中 15 で Qwen3 超え**。これは [[entities/internvl-3|InternVL 3]] の「Native Pre-Training で言語が強くなる」発見が **大型モデルでも成立** することを示す重要な反復実証
  - **LLM ベンダー戦略の継続**: InternVL 3 で Qwen2.5 base に切り替えた戦略を継続、**Qwen3 base に統一**（InternLM3-8B は今回採用されず）。これは Shanghai AI Lab が **「OpenGVLab の役割は視覚側、LLM は最良のもの（Qwen3）を採用」** という分業を確立したことを示す。**OpenAI GPT-OSS の統合** は、自社・Alibaba・OpenAI という 3 系列の LLM を併用する柔軟性を示す
  - **9 サイズ × dense + MoE × Flash の組み合わせ**: 18+ モデルのスイートで、**消費者 GPU（1B/2B）からエンタープライズ MoE（241B）まで** カバー。Flash 効率版を全サイズで提供することで、実用シナリオごとの選択肢を最大化。これは **「研究より実用」** という OpenGVLab の戦略
  - **VSI-Bench で GPT-5 を +32 圧倒の意義**: 空間推論（3D シーン理解）は **物理ロボティクス・自律走行・AR/VR** の基盤能力。InternVL3.5-8B (8B 活性) でも GPT-5 を +18.8 上回るという結果は、**「3D 空間理解はオープンソースが商用を超えた領域」** を示す。embodied AI への大きな前進
  - **既存 wiki ハブとの自然な統合**: [[entities/internvl-3|InternVL 3]] の Native Pre-Training + MPO + VisualPRM の哲学を継承しつつ、Cascade RL（MPO + GSPO）+ MoE + 効率化の 3 軸で前進。**[[entities/mpo|MPO]] が Cascade RL の Stage 1 として進化** という関係性が wiki 内で明示化
  - **InternVL シリーズ全 8 世代の追跡完了**: 1.0 → 1.5 → 2.0 → Mini → 2.5 → 3 → 3.5 が全て wiki 内で追跡可能。MPO（後訓練）+ MMPR（データ）も独立 entity ページがあり、**InternVL エコシステム全体** が wiki で表現される
  - 後続候補: Qwen3 原典 / Qwen2.5-VL / Qwen3-VL（Alibaba の最新 MLLM）/ GLM-4.5V 原典（Zhipu の MLLM）/ Step-3 原典（StepFun）/ Kimi-VL 原典（Moonshot）/ MiMo-VL-RL 原典（Xiaomi）/ GPT-5 関連情報 / GPT-OSS 原典 / GSPO 原典（DeepSeek 関連）/ VisualPRM-v1.1 原典 / Seed1.5-VL（ByteDance）/ UI-TARS（GUI agent）/ DeepSeek-V3 関連 など最新 MLLM 系

## [2026-05-30] ingest | Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond

- 取り込み: `raw/papers/Qwen-VL_ A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond.md`（497 行、130KB、arXiv:2308.12966、Alibaba Group、2023 Aug）
- 作成:
  - [[sources/qwen-vl]] — Qwen-VL シリーズ第 1 世代の要約（Qwen-7B + OpenCLIP ViT-bigG + Position-aware VL Adapter、3 段階学習、`<box>`/`<ref>` 特殊トークン、9.6B、Flickr30K 85.8 / VQAv2 79.5 / RefCOCO 89.36 / TouchStone-Cn 401.2 等）
  - [[translations/qwen-vl]] — Abstract + §1-6 の本文翻訳（References と Appendix A-E は除外、ユーザー指示）。図 1-4 + 表 1-7 を含む。Position-aware VL Adapter、`<img>`/`<box>`/`<ref>` 特殊トークン、[0,1000) 座標正規化、ChatML 対話フォーマット、3 段階訓練の全詳細を含む
  - [[entities/qwen-vl]] — Qwen-VL / Qwen-VL-Chat のエンティティ詳細（9.6B、3 段階学習ハイパラ、4 種特殊トークン、20+ ベンチマーク結果、設計判断のアブレーション要点）
  - `raw/assets/qwen-vl/`: fig1.png（性能比較レーダーチャート）、fig2.png（3 段階学習パイプライン図）、fig3.png（事前学習収束曲線）、fig4.png（クエリ数アブレーション）、demo2.jpg（定性例）、fewshot.png（フューショット比較）の 6 画像をダウンロード
- 更新:
  - [[index]] — Sources（Qwen-VL を InternVL 3.5 の前に配置）、Translations（同上）、Entities Models（Qwen-VL を MMPR の前に配置）の 3 セクションに追加
  - [[log]]
- メモ:
  - **Qwen-VL シリーズ第 1 世代の意義**: Alibaba の Qwen2-VL / Qwen2.5-VL / 将来の Qwen3-VL に至る系譜の **出発点**。2023 年 8 月時点で **「中国語 + 細粒度グラウンディング + OCR」の 3 つを同時に解決した最初のオープンソース MLLM**
  - **「`<box>` / `<ref>` 特殊トークン設計」が MLLM 業界の事実上の標準に**: バウンディング・ボックス座標を **[0, 1000) に正規化した数値文字列**として `<box></box>` で囲む設計は、位置語彙を追加する必要がなく通常のテキスト生成と同じ仕組みでグラウンディングを実現する天才的なシンプルさ。これが後の Qwen2-VL / Qwen2.5-VL のみならず、業界標準として広く採用された
  - **Position-aware VL Adapter の独自性**: BLIP-2 の Q-Former と同じく学習可能クエリ（256 個）で視覚トークンを圧縮するが、**クロスアテンションのクエリ-キー対に 2D 絶対位置エンコーディング**を加える点が独自。これがグラウンディングを可能にする位置情報を保持する鍵。InternVL 系の Pixel Shuffle ([[entities/internvit-300m]]) とは設計思想が異なる
  - **3 段階学習の保守的設計**: Stage1 で LLM 凍結、Stage2 で全凍結解除、Stage3 で ViT 凍結という階層的アプローチ。これは InternVL 3 ([[entities/internvl-3]]) が後に提唱する **Native Multimodal Pre-Training（最初から共学習）** と対照的。当時の主流設計を反映
  - **Flickr30K (0-shot) で 80B の Flamingo を上回る**: 9.6B モデルで Flickr30K 85.8 CIDEr を達成（Flamingo-80B 67.2 を +18.6）。当時のオープンソース MLLM のゼロショット最強クラスを記録
  - **中国語マルチモーダル対話の事実上の標準**: TouchStone Cn 401.2 は当時他モデルが中国語未対応の中で唯一の本格的スコア。VisualGLM、CogVLM など中国語 MLLM の比較基準として 1 年以上機能した
  - **InternVL シリーズとの対比**: 同時期に InternVL ([[entities/internvl]]) も活動を開始し、両者は **2023-2025 年のオープンソース MLLM 2 大系譜**を形成。Qwen-VL は OpenCLIP ViT-bigG + 256 クエリ圧縮 + 中英バイリンガル中心の設計、InternVL は InternViT-6B + Pixel Shuffle + より広範な多言語の設計、と異なる方向性で進化
  - **当時の弱点**: (1) 解像度 448² 固定（後の Qwen2-VL の Naive Dynamic Resolution + M-RoPE で解決）、(2) 256 トークン固定圧縮（高情報密度画像で情報損失、Qwen2-VL の動的トークン数で解決）、(3) Stage1 で LLM 凍結（InternVL 3 の Native Pre-Training とは対照的）、(4) 単一モデル規模のみ（Qwen2-VL で 2B/7B/72B の 3 サイズに拡張）
  - **本 ingest 段階では Qwen-VL は単独 entity として登録**: Qwen2-VL / Qwen2.5-VL は別 entity ページとして将来追加予定。次の ingest で Qwen2-VL や Qwen2.5-VL を取り込めば、Qwen ファミリーの系譜表が完成する
  - **既存 InternVL 系 entities の更新は最小限**: Qwen-VL は InternVL 2.5 / 3 / 3.5 の論文では「対抗・比較対象」として度々言及されるが、InternVL 系の各 entity ページに既に「Qwen-VL を上回る」などの比較が含まれているため、明示的なクロスリンク追加は今回見送り（InternVL 系から見ると Qwen-VL は競合の祖、Qwen-VL から見ると InternVL は同時期の競合系譜）
  - 後続候補: Qwen2-VL / Qwen2.5-VL（次世代の動的解像度 + M-RoPE）/ Qwen3-VL / Qwen3 LLM 原典 / GLM-4.5V / Step-3 / Kimi-VL / MiMo-VL-RL / GPT-OSS 原典 / GSPO 原典 / VisualPRM-v1.1 / Seed1.5-VL / UI-TARS / DeepSeek-V3 / 古典 LLaVA / MiniGPT-4 / BLIP-2 等の歴史的位置付け

## [2026-05-30] ingest | Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution

- 取り込み: `raw/papers/Qwen2-VL_ Enhancing Vision-Language Model's Perception of the World at Any Resolution.md`（669 行、arXiv:2409.12191、Qwen Team Alibaba Group、2024 Sept）
- 作成:
  - [[sources/qwen2-vl]] — Qwen2-VL シリーズ第 2 世代の要約（Naive Dynamic Resolution、M-RoPE、統一画像/動画処理、2B/7B/72B、DFN ViT 共有、Qwen2 LLM、3 段階学習 1.4T トークン、80K 外挿、20 ベンチ超 + 動画 + エージェント + 多言語 OCR の主要結果）
  - [[translations/qwen2-vl]] — Abstract + §1-4 + Acknowledgements の本文翻訳（References と Appendix A は除外、ユーザー指示）。図 1-6 + 表 1-8 を含む
  - [[entities/qwen2-vl]] — Qwen2-VL-2B/7B/72B のエンティティ詳細（モデル詳細、3 主要技術の数式・設定値、3 段階学習、全 22+ ベンチ結果、システム実装、ライセンス・HF パス、限界）
  - `raw/assets/qwen2-vl/`: fig1.jpg（能力レーダーチャート）、fig2.jpg（アーキテクチャ概観）、fig3-mrope.png（M-RoPE 図解、temporal/height/width 分解）、fig4-minpixels.png（min_pixels アブレーション）、fig5-extrapolation.png（80K 外挿性能）、fig6-scale.jpg（モデル・スケーリング）の 6 画像をダウンロード
- 更新:
  - [[index]] — Sources（Qwen2-VL を Qwen-VL の前に配置、降順時系列）、Translations（同上）、Entities Models（同上）の 3 セクションに追加
  - [[entities/qwen-vl]] — 系譜セクションに Qwen2-VL（2024.09、2B/7B/72B、Naive Dynamic Resolution + M-RoPE）の詳細追加
  - [[concepts/foundation-model]] — A. 弱教師あり系セクションに Qwen2-VL を Qwen-VL の後ろに追加（Naive Dynamic Resolution + M-RoPE + DFN ViT）
  - [[concepts/rotary-position-embeddings]] — M-RoPE を導入した代表モデルとして Qwen2-VL を追加
  - [[concepts/weakly-supervised-pretraining]] — Qwen2-VL の弱教師あり Stage1（600B image-text + OCR + classification）を追加
  - [[overview]] — VLM/MLLM 系譜に Qwen2-VL を追加、Qwen 系の系譜（Qwen-VL → Qwen2-VL → Qwen2.5-VL）を明示
  - [[log]]
- メモ:
  - **Qwen-VL シリーズ第 2 世代の意義**: [[entities/qwen-vl|Qwen-VL]] 初代の 3 つの構造的弱点（448² 固定 / 256 トークン固定 / 動画は 2D 画像系列）を一気に解消。**Naive Dynamic Resolution + M-RoPE は MLLM 業界の事実上の標準**として広く採用された
  - **「タイル分割 vs ViT 自体の任意解像度対応」の路線対立**: 同時期の [[entities/internvl-1-5|InternVL 1.5]]（2024 Apr）が **35 通りのアスペクト比 × 1-40 タイル分割 + Pixel Shuffle 1/4 圧縮**で動的解像度を実装したのに対し、Qwen2-VL は **「ViT の絶対位置埋め込みを 2D-RoPE に置換」**でより根源的に任意解像度対応。Qwen2-VL のアプローチがその後の業界標準になった点が技術史的に重要
  - **M-RoPE の設計の天才性**: temporal/height/width の 3 成分に分解することで、(1) テキスト処理時は 1D-RoPE と等価、(2) 画像処理時は時間軸を固定して空間 2 軸のみ使用、(3) 動画処理時は時間軸が増分。**副次効果として「位置 ID 値が小さく抑えられて長文脈外挿に有利」**。学習 16K で推論 80K まで頑健というのは衝撃的結果
  - **同一 675M ViT を全サイズで共有**: 2B/7B/72B すべてで 675M ViT を共有することで、LLM サイズに関係なく視覚処理コストを一定に保つ。これは Mini-InternVL の InternViT-300M 共有戦略と同じ思想だが、Qwen2-VL は元から 3 サイズで設計されている点が違う
  - **DFN（Apple）ViT を初期化に使う点**: 初代 Qwen-VL の OpenCLIP ViT-bigG（1.9B）からの大幅縮小（1.9B → 0.675B）+ 別系統への切り替え。DFN は Apple の高品質データフィルタリングによる CLIP 系で、Apple の AIMv2 系譜と共通の流れ
  - **3 段階学習 + 累積 1.4T トークン**: Qwen-VL 初代の Stage1 が 1.5B サンプル ≒ 500B トークンだったのに対し約 3 倍。**監督はテキスト・トークンにのみ**という方針は Qwen-VL の伝統を踏襲（cf. InternVL 3 も同じ方針、ただし InternVL 3 は square averaging を追加）
  - **特殊トークンの命名アップデート**: `<box>` → `<|box_start|>`、`<ref>` → `<|object_ref_start|>` 等、ChatML 風の `<|...|>` フォーマットに統一。座標 [0, 1000) 正規化文字列は不変
  - **MMMU 64.5（GPT-4o 69.1 に -4.6）が当時の弱点**: その後 [[entities/internvl-2-5|InternVL 2.5]] が 70.1 で初の 70% 突破、[[entities/internvl-3|InternVL 3]] が 72.2 で再び SOTA、[[entities/internvl-3-5|InternVL 3.5]] が 77.7 でオープンソース新 SOTA と続く。Qwen2-VL は MMMU では追随を後継世代に委ねた形
  - **マルチモーダル化で言語が強くなる発見はまだない**: InternVL 3 が 2025 Apr に発見・主張するこの現象は、Qwen2-VL 段階では論じられていない（言語ベンチ評価が公開されていない）。哲学転換は InternVL 系が先行
  - **GPT-4o を多くの文書 / OCR / 多言語タスクで上回ったオープンソース MLLM**: 2024 年 9 月時点で **DocVQA 96.5 vs 92.8 / OCRBench 877 vs 736 / MTVQA 30.9 vs 27.8 / RealWorldQA 77.8 vs 75.4 / MMVet 74.0 vs 69.1 / MathVista 70.5 vs 63.8** など、商用との差を多数の領域で逆転した重要な瞬間
  - **動画理解の MLLM 標準化**: 20 分以上動画 + Video-MME / EgoSchema / MVBench の公開はその後の MLLM 動画ベンチマーキングの基盤に。EgoSchema 77.9（GPT-4o +5.7）は商用フロンティアを上回る決定的結果
  - **エージェント能力**: 32K 文脈長 + UI 操作（AITZ EM 72.1）+ 関数呼び出し（TM 93.1）+ Number Line/EZPoint 100% は、MLLM をエージェントとして使う実用性を初めて本格的に示した。**ALFRED SR 67.8 で専門モデル ThinkBot を超えた**点も重要（家事ロボット）
  - **VLN（R2R）で -27.3 と専門モデルに大敗**: 3D 空間モデリング・マップ理解は MLLM の弱点として残存。[[entities/internvl-3|InternVL 3]] / [[entities/internvl-3-5|InternVL 3.5]] が VSI-Bench で GPT-4o / GPT-5 を圧倒する一方、ナビゲーションは未解決領域
  - **次の ingest 候補**: Qwen2.5-VL（動的解像度のさらなる強化、動画 frame rate 動的化）/ Qwen3 LLM 原典（多言語強化の基盤）/ DFN 原典（Apple の CLIP 系視覚エンコーダ）/ Qwen2 LLM 原典 / 古典 LLaVA / MiniGPT-4 / BLIP-2 の歴史的位置付け / 古典 Flamingo（Qwen2-VL の比較対象）

## [2026-05-30] ingest | Qwen2.5-VL Technical Report

- 取り込み: `raw/papers/Qwen2.5-VL Technical Report.pdf`（23 ページ PDF、arXiv:2502.13923、Qwen Team Alibaba Group、2025 Feb 19）
- 作成:
  - [[sources/qwen2-5-vl]] — Qwen2.5-VL シリーズ第 3 世代の要約（Window Attention + ViT from scratch、MRoPE absolute time、動的 FPS、QwenVL HTML format、3B/7B/72B、4.1T トークン、3 段階事前学習 + 2 段階事後学習、GUI agent 商用級進化、20+ ベンチ + 動画 + GUI エージェントの主要結果）
  - [[translations/qwen2-5-vl]] — Abstract + §1-5 の本文翻訳（References pp.16-23 は除外、Appendix なし、ユーザー指示）。図 1 + 表 1-9 を含む
  - [[entities/qwen2-5-vl]] — Qwen2.5-VL-3B/7B/72B のエンティティ詳細（モデル詳細、4 主要技術の数式・設定値、3 段階事前学習 + 2 段階事後学習、全 30+ ベンチ結果、QwenVL HTML フォーマット、限界）
  - `raw/assets/qwen2-5-vl/fig1.png`: ユーザーが手動で `raw/images/fig1.png` に置いた Qwen2.5-VL フレームワーク図（ViT + Window/Full Attention + Vision Encoder + Qwen2.5 LM Decoder + 動的 FPS サンプリング + Conv3D の構造図）を、CLAUDE.md スキーマ §7 に従い `raw/assets/qwen2-5-vl/fig1.png` に再配置
- 更新:
  - [[index]] — Sources（Qwen2.5-VL を Qwen2-VL の前に配置、降順時系列）、Translations（同上）、Entities Models（同上）の 3 セクションに追加
  - [[entities/qwen2-vl]] — 系譜セクションに Qwen2.5-VL（2025.02、3B/7B/72B、Window Attention + MRoPE absolute time + 4.1T + QwenVL HTML + GUI Agent）の詳細追加
  - [[concepts/foundation-model]] — A. 弱教師あり系セクションに Qwen2.5-VL を Qwen2-VL の前に追加
  - [[concepts/rotary-position-embeddings]] — 派生に「MRoPE Aligned to Absolute Time」を追加
  - [[concepts/weakly-supervised-pretraining]] — Qwen2.5-VL の 3 段階事前学習（4.1T トークン、Visual 1.5T + Multimodal 2T + Long-Context 0.6T）を追加
  - [[concepts/zero-shot-transfer]] — オープン語彙検出 ODinW、点グラウンディング、CountBench でのゼロショット転移を追加
  - [[overview]] — VLM/MLLM 系譜に Qwen2.5-VL を追加、Qwen 系の系譜（Qwen-VL → Qwen2-VL → Qwen2.5-VL）を完成
  - [[log]]
- メモ:
  - **Qwen-VL シリーズ第 3 世代の意義**: [[entities/qwen2-vl|Qwen2-VL]] が「任意解像度」を実現したのに続き、Qwen2.5-VL は (i) **計算複雑性問題（Window Attention で線形化）**、(ii) **動画の時間動態（MRoPE absolute time で FPS 非依存）**、(iii) **データ規模（1.2T → 4.1T）**、(iv) **GUI エージェント（ScreenSpot Pro 1.6% → 43.6% = 27× 飛躍）** という 4 つの未解決課題を一気に解決
  - **「視覚言語モデルから視覚エージェントへの転換点」**: 単に MLLM の精度を上げたのではなく、**ScreenSpot Pro / Android Control / AndroidWorld / MobileMiniWob++ で GPT-4o / Gemini 2.0 / Claude を圧倒**するレベルに到達。これは「実用ツールとしての MLLM」の新時代の始まりを示す。OSWorld（デスクトップ）のみ Claude に後れる点が興味深い
  - **MRoPE Aligned to Absolute Time の天才性**: Qwen2-VL の MRoPE は temporal ID をフレーム番号に結びつけていたため、30 fps 動画と 5 fps 動画で「10 秒」の表現が違っていた。Qwen2.5-VL は **temporal ID 間の間隔を絶対時間（秒）に揃える**だけで、追加ヘッドやテキスト・タイムスタンプ注入なしに時間グラウンディングを実現。**Charades-STA mIoU 50.9 で GPT-4o 35.7 を +15.2 圧倒**という結果が決定的
  - **Window Attention + ViT from scratch の決断**: Qwen2-VL は DFN（Apple）の ViT を初期化に使っていたが、Qwen2.5-VL は **DataComp + 社内データでゼロから学習**。これは「他社の CLIP/DFN に依存しない自社視覚エンコーダ」を確立する戦略的決断。Window Attention 採用（32 層中 4 層のみ完全自己注意）で計算コストを削減しつつ、native resolution のままで処理
  - **QwenVL HTML フォーマットの革新**: 文書解析を「複数の専門モデルの組み合わせ」から「単一汎用モデル + 統一フォーマット」へ転換。data-bbox 属性で座標、HTML タグでレイアウト・表・チャート・数式・楽譜（ABC notation）・化学式（SMILES）を表現。**OCRBench_v2 で Gemini 1.5-Pro を英語 +9.6 / 中国語 +20.6 ポイント圧倒**の決定打。**ABC notation と SMILES** という業界標準を採用した点が玄人受け
  - **MMMU で初めて 70 突破**: Qwen2-VL 64.5 → Qwen2.5-VL-72B 70.2（+5.7）で初の 70 突破、GPT-4o 69.1 / Claude-3.5-Sonnet 68.3 / InternVL2.5-78B 70.1 と肩を並べる。同時期に [[entities/internvl-2-5|InternVL 2.5]] も 70.1 で 70% 突破していたため、**2024 年末〜2025 年初頭は MLLM が「70% の壁」を突破した記念碑的時期**
  - **マルチモーダル化で言語が強くなる現象**: Qwen2.5-VL-72B は **純粋テキスト・タスクで Qwen2.5-72B（純粋 LLM）とほぼ同等**を維持。LiveBench / MultiPL-E / IFEval ではむしろ上回る。これは [[entities/internvl-3|InternVL 3]] が 2025 Apr に発見・主張する現象が **2025 Feb の Qwen2.5-VL でも既に観察されていた**証拠
  - **3B サイズの登場（2B → 3B）**: Qwen2-VL の最小モデルは 2B だったが、Qwen2.5-VL は 3B が最小。Qwen2.5 LLM ファミリーに 2B が存在しないため。これにより最小モデルがやや大型化、エッジ AI の実用性に微妙な影響
  - **モデル間互換性の保持**: Qwen2-VL から Qwen2.5-VL への移行で、`<box>` / `<ref>` 特殊トークンや MRoPE 基本構造は維持されつつ、Window Attention や MRoPE absolute time が追加。アーキテクチャ的に大きな破断なし、段階的進化
  - **InternVL 系と Qwen 系の路線対立の継続**: [[entities/internvl-3|InternVL 3]]（2025.04）が **Native Multimodal Pre-Training** を提唱したのに対し、Qwen2.5-VL（2025.02）は依然として **「LLM 事前学習 → MLLM 適応」のパラダイム**を維持。両系列の哲学的対立が続く
  - **ScreenSpot Pro の +42 ポイント飛躍**: 高解像度プロ向け UI ベンチである ScreenSpot Pro で、Qwen2-VL の 1.6% から Qwen2.5-VL の 43.6% へ **+42 ポイント（27×）の飛躍**。これは GUI 専用エージェント Aguvis-72B（23.6）すら大きく上回る結果。**「Window Attention で高解像度を効率処理」+「絶対座標グラウンディング・データの大規模化」+「Agent Data の充実」** の効果が複合的に現れた
  - **訓練データの公開度差**: [[entities/internvl-3|InternVL 3]] が `OpenGVLab/InternVL-Data` で完全公開しているのに対し、Qwen2.5-VL は訓練データ非公開（モデル重みは公開）。オープン・サイエンスの観点では InternVL 系に劣後
  - **PDF からの ingest という新しい運用**: 本論文は PDF 形式で、Web Clipper による markdown 化前に直接 ingest。図はユーザーが手動で `raw/images/fig1.png` に配置 → Claude が `raw/assets/qwen2-5-vl/` に再配置するワークフロー。今後の PDF 論文 ingest の標準フローとなる
  - **Appendix なしの論文構造**: Qwen2.5-VL Technical Report は §1-5（Authors）まで本文 + References のみで Appendix なし。Qwen-VL 初代と Qwen2-VL には Appendix があったが、Qwen2.5-VL では本文に統合された
  - **次の ingest 候補**: Qwen3-VL（Qwen3 LLM ベースの後継、まだ未公開？）/ DFN 原典（Apple の CLIP 系視覚エンコーダ）/ Qwen2.5 LLM 原典 / GLM-4.5V / Step-3 / Kimi-VL / Molmo（PixMo データセットの原典）/ Gemini 2.0 関連 / Aguvis 原典（GUI エージェント比較対象）/ AndroidWorld 原典 / OSWorld 原典 / 古典 LLaVA / MiniGPT-4 / BLIP-2

## [2026-05-30] ingest | Qwen3-VL Technical Report

- 取り込み: `raw/papers/Qwen3-VL Technical Report.pdf`（26 ページ PDF、arXiv:2511.21631v2、Qwen Team Alibaba Group、2025 Nov 27）
- 作成:
  - [[sources/qwen3-vl]] — Qwen3-VL シリーズ第 4 世代の要約（Interleaved MRoPE + DeepStack + テキスト・ベース時間整合の 3 構造革新、6 サイズ × 2 バリアント = 12 公開モデル、256K ネイティブ + 1M 外挿、4 段階事前学習 ~2.2T、Strong-to-Weak Distillation + SAPO RL + Thinking with Images、商用最先端との対比結果、Apache 2.0）
  - [[translations/qwen3-vl]] — Abstract + §1-7（References p.26+ は除外、Appendix なし、ユーザー指示）。図 1-3 + 表 1-12 を含む
  - [[entities/qwen3-vl]] — Qwen3-VL 6 サイズ × 2 バリアント = 12 モデルのエンティティ詳細（モデル詳細、3 構造革新の数式・設定値、4 段階事前学習 + 3 段階事後学習 + Thinking with Images、全 50+ ベンチ結果、QwenVL-HTML + QwenVL-Markdown、Needle-in-a-Haystack、限界）
  - `raw/assets/qwen3-vl/`: fig1.png（フレームワーク図、Vision Encoder + DeepStack + Qwen3 LM Dense/MoE Decoder + Native Resolution Input + Conv3D）、fig2-multilingual-ocr.png（39 言語の OCR 精度棒グラフ、32 言語で 70% 超）、fig3-needle.png（Needle-in-a-Haystack ヒートマップ、256K で完全 100% / 1M で 99.5%）の 3 画像を `raw/images/` から再配置（CLAUDE.md §7 準拠）
- 更新:
  - [[index]] — Sources（Qwen3-VL を Qwen2.5-VL の前に配置、降順時系列）、Translations（同上）、Entities Models（同上）の 3 セクションに追加
  - [[entities/qwen2-5-vl]] — 系譜セクションに Qwen3-VL（2025.11、6 サイズ × 2 バリアント = 12 モデル、Interleaved MRoPE + DeepStack + テキスト・タイムスタンプ + MoE + thinking モード）の詳細追加
  - [[concepts/foundation-model]] — A. 弱教師あり系セクションに Qwen3-VL を Qwen2.5-VL の前に追加
  - [[concepts/rotary-position-embeddings]] — 派生に「Interleaved MRoPE」を追加、関連ページに Qwen3-VL 追加
  - [[concepts/weakly-supervised-pretraining]] — Qwen3-VL の 4 段階事前学習（S0 Merger 67B + S1/S2 各 1T + S3 100B 262K）を追加
  - [[concepts/zero-shot-transfer]] — 39 言語 OCR、9-DoF 3D グラウンディング、GUI エージェント（ScreenSpot Pro/OSWorld/AndroidWorld）でのゼロショット転移を追加
  - [[overview]] — VLM/MLLM 系譜に Qwen3-VL を追加、Qwen 系の系譜（Qwen-VL → Qwen2-VL → Qwen2.5-VL → Qwen3-VL）を完成
  - [[log]]
- メモ:
  - **Qwen-VL シリーズ第 4 世代の意義**: [[entities/qwen-vl|Qwen-VL]] (`<box>`/`<ref>` + 256 固定) → [[entities/qwen2-vl|Qwen2-VL]] (Naive Dynamic Resolution + M-RoPE) → [[entities/qwen2-5-vl|Qwen2.5-VL]] (Window Attention + MRoPE absolute time + QwenVL HTML) → Qwen3-VL (Interleaved MRoPE + DeepStack + テキスト・タイムスタンプ + MoE + thinking) という Qwen 系の完成形。商用 GPT-5 / Gemini 2.5-Pro / Claude Opus 4.1 と多くのベンチで対等以上
  - **「位置エンコーディングで時間を表現」から「テキスト・トークンで時間を表現」への転換**: Qwen2.5-VL の MRoPE absolute time は temporal ID を絶対秒数に紐付けたが、長動画で temporal ID が過度に大きく疎になる問題が判明。Qwen3-VL は MRoPE absolute time を**完全に捨て**、`<3.0 seconds>` のような明示的テキスト・タイムスタンプ・トークンに置換。**Charades-STA mIoU 64.8（Qwen2.5-VL 50.9 から +13.9）** という劇的改善が決定打。これは「シンプルな解決策が往々にして勝つ」という Occam の剃刀の好例
  - **Interleaved MRoPE の発見**: Qwen2-VL から続いていた「temporal/height/width を埋め込み次元の塊に分割」する MRoPE は周波数スペクトル不均衡を生むことが Huang et al., 2025 で示され、Qwen3-VL は t/h/w を埋め込み次元にわたって**均一に交互配置**する設計に変更。これは MRoPE 路線の自然な発展で、後続の動画 MLLM で標準採用される可能性が高い
  - **DeepStack の MLLM への持ち込み**: Meng et al., 2024 の DeepStack（マルチスケール視覚入力からトークンを積み重ね）を **ViT 中間層からトークンを抽出する形に拡張**して LLM の最初 3 層に注入。**追加文脈長を導入しない多層融合**という方針が新しい。アブレーション（200B トークン学習、表 12）で平均 +1.3 ポイント、特に InfoVQA +2.3 / DocVQA +1.6 で細粒度視覚理解に効く
  - **SigLIP-2 から継続学習への戦略転換**: Qwen2.5-VL は ViT をゼロから学習する野心的方針だったが、Qwen3-VL は **SigLIP-2 の事前学習済みチェックポイントから初期化して継続学習**に戻った。これは「自前 ViT vs 既存 VFM」の戦略の揺れ動きで、Qwen3-ViT が SigLIP-2 を OmniBench で +8.6 ポイント上回る（CLIP 段階）という結果でこの戦略を正当化
  - **MoE スケーリング 235B-A22B の到達**: [[entities/internvl-3-5|InternVL 3.5]] の 241B-A28B（2025 Aug）に続き、Qwen3-VL も **235B-A22B フラッグシップ MoE** を投入。両方とも 2025 年後半に MLLM が「200B 級 MoE」時代に突入したことを示す。**MoE 30B-A3B** という中規模 MoE が Gemini 2.5 Flash / GPT-5 mini を凌駕することで、MoE 効率モデルが商用フロンティアと完全に互角になった
  - **Thinking モードの本格採用**: OpenAI o1（2024.09）以降 LLM 業界の標準となった「長 CoT 推論モデル」が、Qwen3-VL で正式に MLLM にも導入。各サイズに **Instruct（non-thinking）と Thinking** の 2 バリアントを公開する設計は実用性の高い選択。特に thinking モデルが MathVista 85.8 / MathVision 74.6 で SOTA を達成
  - **256K ネイティブ + 1M 外挿**: Qwen2.5-VL の 32K（256K は post-training で拡張）から、Qwen3-VL は **256K がネイティブ**になり、YaRN で 1M トークン = **2 時間動画**まで外挿可能。Needle-in-a-Haystack で 256K 完全 100% / 1M で 99.5% は、長文脈耐性の新ベンチマーク
  - **GUI エージェントの劇的進化**: ScreenSpot Pro 43.6 → **62.0** (+18.4)、OSWorld 8.83 → **38.1** (4.3× 飛躍)、AndroidWorld 35% → **63.7%** (+28.7) と、わずか 9 ヶ月で GUI エージェント能力が劇的に伸びた。**OSWorld で Claude Opus 4 (44.4) に -6.3** にまで肉薄、デスクトップ・エージェント領域でも商用と互角に近づく
  - **マルチモーダル化で言語が強くなる現象の継続実証**: Qwen3-VL-235B-A22B-Instruct は Qwen3-235B-A22B-Instruct-2507（純粋 LLM）に対して AIME-25 +4.4 / HMMT-25 +2.0 / LiveCodeBench v6 +2.5 と推論ベンチで上回る。[[entities/internvl-3|InternVL 3]] / [[entities/qwen2-5-vl|Qwen2.5-VL]] で観察された現象が大型 MoE モデルでも継続。**「マルチモーダル学習は言語能力を強化する」が業界共通認識に**
  - **多言語 OCR 10 → 39 言語**: Qwen2.5-VL の 10 言語（中英 + 仏・独・伊・西・葡・露・日・韓・越・アラ）から **29 言語追加で 39 言語**に拡張。スワヒリ語・ヘブライ語・チェコ語・ウルドゥー語・タイ語・スウェーデン語・セルビア語・デンマーク語など。32/39 言語で 70% 超の OCR 精度
  - **座標系の [0, 1000] 正規化への復帰**: Qwen2.5-VL は **絶対座標**に切り替えたが、Qwen3-VL は **[0, 1000] 正規化座標系**に戻った。これは「解像度とアスペクト比の変動への頑健性」と「後処理の単純化」を優先した設計判断。Qwen-VL 初代の方針への回帰
  - **InternVL 系と Qwen 系の路線対立**: [[entities/internvl-3|InternVL 3]] / [[entities/internvl-3-5|InternVL 3.5]] が **Native Multimodal Pre-Training + Cascade RL + ViR + DvD** という独自路線を歩む一方、Qwen3-VL は **「LLM 事前学習 → MLLM 適応」のパラダイム**を維持しつつ Interleaved MRoPE + DeepStack + テキスト・タイムスタンプで応酬。**両系列が完全に異なる技術選択で同等の性能に到達**している点が技術史的に興味深い
  - **訓練データ非公開**: [[entities/internvl-3|InternVL 3]] が `OpenGVLab/InternVL-Data` で完全公開しているのに対し、Qwen3-VL は訓練データ非公開（モデル重みは Apache 2.0）。オープン・サイエンスの観点では InternVL 系に劣後するが、商用利用性の点で Apache 2.0 は強力
  - **Qwen3 LLM ベースへの完全移行**: Qwen2-VL は Qwen2、Qwen2.5-VL は Qwen2.5、Qwen3-VL は Qwen3 と、Qwen LLM 世代と完全に同期。**LLM 自体の進化が MLLM の進化を牽引**するという Qwen ファミリーの戦略が一貫
  - **「視覚エージェント + 思考モデル + MoE」の 3 軸統合**: 単一論文で (i) 推論モデル（thinking）、(ii) MoE スケーリング、(iii) GUI エージェントの 3 つの 2025 年の主要トレンドすべてを統合。**業界最先端の集大成**としての位置付けが強い
  - **PDF からの ingest 運用 2 回目**: Qwen2.5-VL に続き Qwen3-VL も PDF 形式で直接 ingest。図 3 枚（fig1 / fig2-multilingual-ocr / fig3-needle）を `raw/images/` から `raw/assets/qwen3-vl/` に再配置するワークフローが確立
  - **次の ingest 候補**: Qwen3 LLM 原典 / DeepStack 原典（Meng et al., 2024）/ SAPO 原典（Gao et al., 2025）/ CoMP 原典（Chen et al., 2025）/ SigLIP-2 原典（[[entities/siglip]] と関連、Tschannen et al., 2025）/ Omni3D 原典（Brazil et al., 2023）/ Molmo + PixMo（Allen AI）/ Gemini 2.5 関連 / GPT-5 関連 / Claude Opus 4.1 / GLM-4.5V / Step-3 / Kimi-VL / Aguvis / AndroidWorld / OSWorld 原典 / 古典 LLaVA / MiniGPT-4 / BLIP-2

## [2026-05-30] ingest | Qwen3.5-Omni Technical Report

- 取り込み: `raw/papers/Qwen3.5-Omni Technical Report.md`（579 行、Web Clipper 形式 markdown、Qwen Team Alibaba Group、2026、arXiv:2604.15804）
- 作成:
  - [[sources/qwen3-5-omni]] — Qwen-Omni シリーズ最新版の要約（Thinker-Talker 構造 + Hybrid MoE Thinker/Talker + AuT + ARIA + 256K + Audio-Visual Vibe Coding、Plus/Flash 2 バリアント、215 音声・AV サブタスク SOTA、Gemini-3.1 Pro 凌駕、Qwen ファミリーの Omni 統合路線）
  - [[translations/qwen3-5-omni]] — Abstract + §1-8（References 除外）の本文 + 付録翻訳。図 1-3 + 表 1-15 を含む（付録の多言語 ASR/翻訳全 60 言語テーブルも含む）
  - [[entities/qwen3-5-omni]] — Qwen3.5-Omni-Plus / Qwen3.5-Omni-Flash のエンティティ詳細（モジュール構成、7 主要進化、3 段階事前学習 + Thinker 3 段階 + Talker 4 段階の事後学習、First-Packet 遅延、全ベンチ結果、限界、Qwen ファミリーの 2 主軸）
  - `raw/assets/qwen3-5-omni/`: fig1-overview.png（unified end-to-end model 概観）、fig2-architecture.jpg（Thinker-Talker + MoE + AuT + Vision Encoder + MTP + Streaming Codec Decoder の全体構造）、fig3-aut.png（AuT 概観、4000 万時間学習、6.25Hz）の 3 画像を ar5iv（arxiv:2604.15804）からダウンロード
- 更新:
  - [[index]] — Sources（Qwen3.5-Omni を Qwen3-VL の前に配置）、Translations（同上）、Entities Models（同上）の 3 セクションに追加
  - [[entities/qwen3-vl]] — Qwen ファミリーの 2 主軸を明示するため、関連ページに Qwen3.5-Omni（並列発展する Omni 統合路線）を追加
  - [[overview]] — VLM/MLLM 系譜とは別軸の Omnimodal 系として Qwen-Omni 系譜（Qwen2.5-Omni → Qwen3-Omni → Qwen3.5-Omni）を新規セクションで追加
  - [[log]]
- メモ:
  - **Qwen ファミリーの 2 つの主軸の確立**: VL 純化路線（[[entities/qwen3-vl|Qwen3-VL]]: 画像・動画・文書・GUI エージェント）と Omni 統合路線（Qwen3.5-Omni: テキスト + 画像 + 音声 + 動画 + 音声-視覚 + 音声生成）が並列発展。同じ Qwen3.5 LLM ベースを共有しつつ、異なるユースケースに最適化された 2 系列が完成
  - **Thinker-Talker 構造の意義**: テキスト生成（Thinker）と音声生成（Talker）を **別個の MoE モデル**として分離し、Talker が Thinker の高レベル表現を直接受け取る設計。これは GPT-4o の単一モデル直接出力（仮想）や Gemini の dual-model 結合とも異なる、Qwen 独自の omnimodal アーキテクチャ
  - **AuT のゼロから学習という決断**: SigLIP-2（視覚）の継続学習路線とは異なり、音声エンコーダ AuT は **ゼロから 4000 万時間音声で学習**。これは Whisper / SeamlessM4T のような既存音声基盤モデルを使わず、Qwen-ASR が生成した独自データセットで自社音声 ViT を構築する戦略。6.25Hz トークンレート（フレームあたり 160ms）は商用フロンティアでは非常に高い圧縮率
  - **ARIA の天才性**: テキスト-音声トークン化レート不一致による単語スキップ・誤発音・数字レンダリング曖昧化は、Whisper / Sea Tongue / GPT-4o-audio などすべてのストリーミング音声 LLM の共通問題。ARIA の **適応的レート制約（累積音声-テキスト比率がグローバル比率を超えない）** は MFA の事前整合や固定交互配置のような硬直したアプローチを置換する優雅な解決。**Qwen3.5-Omni の核心的貢献の 1 つ**
  - **Audio-Visual Vibe Coding の出現**: 音声-視覚命令から直接実行可能コードを生成する能力が **omnimodal モデルで自然に出現**。これは GPT-4o の画面共有 → コード生成のような単一モダリティ命令とは異なり、**音声と視覚の同時入力**から具体的コードへの跳躍。Embodied AI の重要な前駆体
  - **共同 video-audio 学習の効果**: Qwen3.5-Omni-Plus は同サイズ純粋 VL の Qwen3.5-Plus-Instruct を **動画 6/6 ベンチで上回る**（VideoMME, MLVU, MVBench, LVBench, MMVU, MME-VideoOCR）+ **RealWorldQA で +5.0**。これは「視覚と聴覚は本質的に結合している」という人間的知覚モデルを実証
  - **256K + 10 時間音声 + 400 秒動画**: [[entities/qwen3-vl|Qwen3-VL]] の 256K + YaRN 1M（2 時間動画）と異なり、Qwen3.5-Omni は **ネイティブで 10 時間音声 + 400 秒 720P 動画**をサポート。実世界の会議録音・講義動画・長尺コンテンツ理解に直接適用可能
  - **first-packet 遅延 235ms (Flash)**: リアルタイム音声対話の体感上の境界（人間の応答遅延 ~200-300ms）に到達。これは **「人間と区別不能な対話速度」** を意味し、Qwen3.5-Omni が実用 omnimodal エージェントの実用域に入ったことを示す
  - **多言語サポートの圧倒的網羅**: 音声入力 113 言語+方言（74 言語 + **39 中国方言**、客家語・閩南語・温州方言・四川語等を含む）/ 音声出力 36 / テキスト 201。これは GPT-4o（~50 言語）/ Gemini-3.1 Pro（~100 言語）を大きく上回り、特に **中国方言の網羅**は中国市場での圧倒的優位性を意味する。Kespeech 中国方言 ASR で Gemini の 6.8× 改善が決定打
  - **API のみ公開という戦略変更**: [[entities/qwen3-vl|Qwen3-VL]] が Apache 2.0 で全モデル重みを公開したのに対し、Qwen3.5-Omni は **API のみ**。これは音声合成の悪用懸念（ディープフェイク、なりすまし）や、4000 万時間音声学習の独自データ保護を反映した可能性。**Qwen ファミリーで初の API 限定モデル**
  - **arXiv 番号 2604.15804 と仮想性**: 番号の「2604」は 2026 年 4 月を示唆、Gemini-3.1 Pro / GPT-5 / Claude Opus 4.1 等のベースラインも仮想的。これは Web Clipper で取得した markdown が arxiv の公式番号体系を超える未来形式である可能性、または論文自体が技術ロードマップ的位置付けである可能性を示唆
  - **次の ingest 候補**: Qwen2.5-Omni 原典（Thinker-Talker 構造の起点）/ Qwen3-Omni-30B-A3B 原典（MoE 初導入）/ Qwen3.5 LLM 原典 / DeepStack 原典 / Whisper 原典（AuT との対比）/ SeamlessM4T（多言語音声基盤の対比）/ AudioPaLM / VALL-E（音声生成系）/ Gemini 2.5 関連 / GPT-5 関連 / Claude Opus 4.1 / OmniCloze 原典 / OmniGAIA 原典 / VoiceBench 原典

## [2026-05-30] query | ViT における解像度処理の進化（固定 → 任意解像度）

- 質問: 「ViT などでは固定解像度のものが多いと思うが、なぜ固定解像度が多いのか。任意解像度で入力できるようになったのはなぜか。ViT の登場からの解像度の扱いの変化は？」
- 作成:
  - [[questions/vit-dynamic-resolution-evolution]] — wiki 初の question ページ。固定解像度の 4 原因（学習可能絶対位置埋め込み / 事前学習データ規格化 / O(n²) 計算 / 帰納バイアス弱）+ 任意解像度化の 3 圧力（MLLM タスク要求 / RoPE / ハードウェア成熟）+ 進化 4 フェーズ年表（固定 2020-2023 / タイル分割 2024 / ViT 改造 2024.09- / SSL 並行進化）+ タイル路線 vs ViT 改造路線の対立 + ViT 初期化戦略の揺れ動き（DFN → ゼロから → SigLIP-2 継続学習）+ まとめ図解
- 更新:
  - [[index]] — Questions セクションを「（まだありません）」から本ページのエントリに置換
  - [[log]]
- 参照ページ（既存）:
  - [[concepts/vision-transformer]] — ViT の基礎、固定解像度の根本理由
  - [[concepts/rotary-position-embeddings]] — RoPE 系の進化
  - [[entities/qwen-vl]] / [[entities/internvl-1-5]] / [[entities/qwen2-vl]] / [[entities/qwen2-5-vl]] / [[entities/internvl-3]] / [[entities/internvl-3-5]] / [[entities/qwen3-vl]] / [[entities/dinov3]]
- メモ:
  - **既存 wiki ハブの自然な統合**: 質問への回答が、既に ingest 済みの Qwen-VL 系 4 世代 + InternVL 1.5 / 2.5 / 3 / 3.5 + DINOv3 + ViT 概念ページ + RoPE 概念ページを **時系列・路線対立軸で横断的に整理**する形になった。**wiki が「読み・書き・更新する側」としての LLM の能力を実証する好例**
  - **「タイル分割路線 vs ViT 改造路線」の対立軸**: 2024 年に並走した 2 つの解像度処理アプローチを明確化。InternVL 1.5 / 2.5 / Mini-InternVL が前者、Qwen2-VL / 2.5-VL / 3-VL / DINOv3 が後者。InternVL 3.5 の ViR は「動的圧縮率」という第 3 路線とも言える
  - **「ViT 初期化戦略の揺れ動き」の発見**: Qwen2-VL（DFN）→ Qwen2.5-VL（ゼロから）→ Qwen3-VL（SigLIP-2 継続学習）という揺り戻しが、wiki 内の既存記述から明確に浮かび上がった。これは ingest 時には個別に記録していたが、質問の回答で初めて **「揺れ動き」という上位構造**として明示化された
  - **「学べる教訓」セクションの追加**: 単なる年表記述に留まらず、5 つの抽象的教訓（構造的可能性と実装制約の分離 / 数学的進化がパラダイム転換を可能に / タスク要求が技術進化を駆動 / 2 路線並行発展 / 初期化戦略の循環）を抽出。これは [[concepts/foundation-model]] や [[concepts/weakly-supervised-pretraining]] と並ぶ「メタ知識ページ」としての価値
  - **質問ページが新しい接続を生む**: 単なる Q&A ではなく、wiki 内の **異なる時期にingest された複数モデルを「解像度」という軸で再編成**するメタページ。これは Karpathy 流 LLM Wiki の理念（「LLM が読み・書き・更新する」）の自然な発展形
  - **今後の question ページ候補**: 「Qwen 系列 vs InternVL 系列の哲学的対立」「2024-2025 年の MLLM ベンチマーク水準の変化」「視覚エージェントの進化（AITZ → ScreenSpot Pro → OSWorld）」「動画理解における時間表現の歴史（TM-RoPE → MRoPE absolute time → テキスト・タイムスタンプ）」「事前学習データ規模の競争（CLIP 4億 → Qwen2-VL 1.4T → Qwen3-VL 2.2T）」など、wiki 内の横断的整理が可能な質問が多数

## [2026-05-30] lint | 全 wiki 健康診断と修正

- 点検結果: **148 ページ規模で dangling link 3 件、孤立ページ 0、スキーマ違反 0** という非常に健全な状態
- 検出された 3 dangling リンクと 2 古い記述、計 5 件をすべて修正 + 2 新規ページ作成
- 修正:
  - **A2 修正**: [[log]] L874 の `[[entities/siglip-2]]` を `[[entities/siglip\|SigLIP 2]]` へ（[[entities/siglip]] が SigLIP 2 を内包、25 箇所言及・§SigLIP 2 セクションあり）
  - **A3 修正**: [[sources/qwen-vl]] の `[[native-multimodal-pre-training]]` を `[[entities/internvl-3\|Native Multimodal Pre-Training]]` へ
  - **B1 修正**: [[entities/qwen2-5-vl]] 系譜セクションの「Qwen3-VL（2025-）← Qwen3 LLM ベースの後継（将来追加予定）」を [[entities/qwen3-vl]] の既追加リンク + [[entities/qwen3-5-omni]] 並列発展への言及に置換
  - **B2 修正**: [[entities/qwen-vl]] 系譜セクションの「Qwen2.5-VL（2025）（wiki 未追加）」「Qwen3-VL（2025-）（wiki 未追加）」を [[entities/qwen2-5-vl]] / [[entities/qwen3-vl]] / [[entities/qwen3-5-omni]] の既追加リンクに置換
- 作成:
  - **C1 修正**: [[entities/internvl-2]] — InternVL 2.0（2024.07）の概要エンティティ・ページ。**公式 Technical Report リリースなし世代**だが、後続 [[sources/mini-internvl]] / [[sources/internvl-2-5]] / [[sources/internvl-3]] で頻繁にベースライン引用されるため、シリーズ系譜の歯抜けを補完。1B〜76B の 7 サイズスイート初確立、InternVL2-Llama3-76B フラッグシップ、MMMU 62.7
  - **A1 修正**: [[entities/moco]] — MoCo（FAIR, CVPR 2020）の概要エンティティ・ページ。[[entities/byol]] / [[entities/simclr]] / [[sources/byol]] から参照される **momentum encoder + queue の SSL 対比学習の起点**。v1 (2019) → v2 (2020) → v3 (2021、ViT 対応) の進化、[[entities/byol]] の target network / [[entities/dino]] の teacher の直接の源流であることを明示
- 更新:
  - [[index]] — Entities Models に [[entities/internvl-2]]（Mini-InternVL の前）と [[entities/moco]]（BYOL の前）を追加
  - [[log]]
- 残課題（許容範囲内）:
  - **InternVL 2.0 公式 Technical Report の ingest 候補**: HuggingFace モデルカード + ブログ記事ベースの本ページから、公式論文形式の ingest へ昇格できれば完璧。ただし InternVL 2.0 は技術報告なし世代のため、現在の概要ページで十分機能
  - **MoCo 原典の ingest 候補**: He et al., CVPR 2020 ("Momentum Contrast for Unsupervised Visual Representation Learning") の専用 source ページ作成は次の SSL 系 ingest 候補。MoCo v2 / v3 まで含めると 3 論文系列
- メモ:
  - **「wiki 健康度」の高さ**: 148 ページ規模で純粋な dangling リンクは 3 件のみ。うち 2 件は既存ページへの誤リンク（修正可能）、1 件は ingest 候補マーカー（[[entities/moco]] → 今回作成で解消）。CLAUDE.md §2 のリンク規約「dangling link 許容」が正しく機能している
  - **「未追加」表記の継続更新コスト**: シリーズ系譜セクションで「将来追加予定」「wiki 未追加」と書いたページは、新規 ingest 時に**親世代ページの修正を忘れがち**。Qwen-VL → Qwen2-VL → Qwen2.5-VL → Qwen3-VL → Qwen3.5-Omni の 5 連続 ingest で、初代と 3 世代目の両方でこの問題が発生
  - **対策**: 今後の ingest 時には「親世代の系譜・関連ページセクション」を必ず確認するチェックリスト項目を確立すべき
  - **InternVL 2.0 の wiki 史的役割**: 公式論文がなくても、複数の後続論文から引用される世代は **「概要エンティティ・ページ」として軽量に補完**するのが正しいアプローチ。これは CLAUDE.md §2 の「sources/translations は原典 1 件につき 1 ページ、entities は概念的存在で原典不要」というスキーマと整合
  - **MoCo の SSL 対比学習史での位置**: BYOL / DINO の momentum target / teacher の直接の源流という事実が、3 つの ingest 済みページ（[[sources/byol]] / [[entities/byol]] / [[entities/simclr]]）から参照されるたびに dangling していた。本ページ作成で **SSL 系の系譜表が完成**
  - **次の lint 候補**: 3 ヶ月後の定期点検、または `sources/moco` / `sources/internvl-2` 原典 ingest 後（dangling 防止のためバッククォート表記）

## [2026-05-31] ingest | Gemma 3 Technical Report

- 取り込み: `raw/papers/Gemma 3 Technical Report.md`（903 行、Web Clipper 形式 markdown、arXiv:2503.19786、Gemma Team Google DeepMind、2025 Mar）
- 作成:
  - [[sources/gemma-3]] — Gemma 3 ファミリーの要約（4 サイズ 1B/4B/12B/27B、SigLIP 400M + Pan & Scan + 256 トークン固定圧縮、5:1 local:global attention + sw=1024、QK-norm + GQA、QAT 3 形式、知識蒸留 14T トークン、LMSys Arena Elo 1338、MATH 89.0、Gemini 1.5 Pro 同等、Google 系の wiki 初登録）
  - [[translations/gemma-3]] — Abstract + §1-8 + Appendix 評価セクションの本文翻訳（References と Contributors 名前リスト除外、**ユーザー確認済み: Appendix 含む**）。図 1（zurich-receipt 視覚相互作用）、図 2（能力レーダー）、図 5（KV キャッシュ vs 構成）、図 6（KV キャッシュ vs 文脈長）、図 7（長文脈性能）を `<figure>` で埋め込み。表 1-18 + 評価詳細を含む
  - [[entities/gemma-3]] — Gemma 3 1B/4B/12B/27B × PT/IT のエンティティ詳細（モデル仕様表、4 主要技術、QAT メモリ表、訓練インフラ TPU 数、全主要ベンチ結果、限界、系譜）
  - `raw/assets/gemma-3/`: fig1-zurich-receipt.jpg（視覚相互作用例）、fig2-abilities.png（Gemma 2 vs 3 能力レーダー）、fig5-kv-cache-config.png（KV キャッシュ vs local:global 比設定）、fig6-kv-cache-context.png（KV キャッシュ vs 文脈長）、fig7-long-context.png（RoPE 再スケーリング前後の長文脈性能）の 5 画像を ar5iv からダウンロード
- 更新:
  - [[index]] — Sources（Gemma 3 を Qwen3.5-Omni の前に配置）、Translations（同上）、Entities Models（同上）の 3 セクションに追加
  - [[entities/siglip]] — 「SigLIP を使う応用モデル」セクションに Gemma 3（400M variant を共有 + 凍結、Pan & Scan、PaliGemma 2 凌駕）を追加
  - [[concepts/foundation-model]] — A. 画像-テキスト WSL 系セクションに Gemma 3 を Qwen3-VL の前に追加（Google 系 MLLM 代表として）
  - [[questions/vit-dynamic-resolution-evolution]] — Phase 2 タイル分割路線の表に Gemma 3 P&S を追加、まとめ図解に Google 系最新形として追加、関連ページに追加
  - [[overview]] — 最新更新ライン更新、VLM 系譜に Gemma 3（2025-03、Google DeepMind）を Qwen3-VL の前に追加、WSL 系統リストと ingest 済みリストに Gemma 3 を追加
  - [[log]]
- メモ:
  - **Google 系オープン MLLM の wiki 初登録の意義**: wiki にはこれまで Qwen 系（Alibaba）と InternVL 系（Shanghai AI Lab）の 2 大系譜が ingest 済みだったが、**Google 系 MLLM が欠落**していた。Gemini は商用閉源で ingest 不可、SigLIP は視覚エンコーダのみ。**Gemma 3 が Gemini 2.0 と co-design された Google のオープン MLLM 代表**として登録され、wiki の主要系譜が 3 つに拡張。これは **Qwen 系 / InternVL 系 / Google 系**の三つ巴対立を明確化する重要な追加
  - **LMSys Chatbot Arena Elo 1338 の衝撃**: 27B Dense モデルで Elo 1338 (rank 9) は、**671B/37B MoE の DeepSeek-V3 (1318)、405B Dense の Llama-3.1-405B (1269)、72B Dense の Qwen2.5-72B (1257) を圧倒**。「**より少ないパラメータで商用クラス**」という Google の哲学を実証
  - **タイル分割路線の最新形**: [[questions/vit-dynamic-resolution-evolution]] で整理した「固定 vs 任意解像度」の対立軸において、**LLaVA → InternVL 1.5 → Gemma 3** のタイル分割系譜を完成。**Qwen2-VL の根源的解決（2D-RoPE で ViT 改造）に対し、Google は実装的解決（推論時のみの Pan & Scan）を選択**。両者の哲学的対立が CV 史における重要な対比軸となる
  - **5:1 local:global attention の業界的意義**: 標準 dense transformer の 60% KV キャッシュ・オーバーヘッドを **<15% に削減**。これは長文脈 MLLM の **新標準的 KV キャッシュ削減技術**となる可能性が高い。Qwen3-VL の 256K ネイティブ + 1M YaRN とは異なるアプローチで、**「文脈長を稼ぐ」ではなく「効率を稼ぐ」** 路線
  - **QAT による軽量化の本気度**: 27B モデルを **Int4 で 14.1 GB** に圧縮し、**ハイエンド消費者 GPU（RTX 4090 等）で展開可能**。これは「**研究より実用**」という Google の戦略を象徴。Qwen / InternVL 系の MoE 路線（数十 GB の活性パラメータ）とは対照的な「**物理的軽量化**」アプローチ
  - **SigLIP 400M を共有 + 凍結という保守的設計**: 4B/12B/27B で **同じ視覚エンコーダを共有 + 訓練中完全凍結**。**視覚埋め込みを事前計算**して言語モデル訓練コスト 0 を実現。これは「視覚エンコーダは確立した SigLIP に任せる」という Google の判断で、Qwen2.5-VL の「ViT をゼロから学習」とは正反対の哲学
  - **PaliGemma 2 との関係**: 論文中で **同じ Google 内の PaliGemma 2 を文書理解で +4.4〜+14.4 凌駕** + **4B/12B は 10× 安価転送**と明示。Google 内の VL モデル系列は **PaliGemma（より大規模・専門特化）** と **Gemma 3（軽量・汎用）** の分業体制になっている
  - **マルチモーダル化で言語能力の劣化を回避**: **MATH 89.0** は Gemini 1.5 Pro (86.5) を上回り、Gemini 2.0 Pro (91.8) に肉薄。**マルチモーダル化が言語能力を劣化させない**という証拠で、InternVL 3 / Qwen3-VL の「マルチモーダル化で言語が強くなる」発見と整合
  - **Memorization の劇的低下**: 全 Gemma 3 で前世代より桁違いに低く、個人情報も観察されず。**Apache 2.0 風ライセンスでの公開を安全に行うための徹底**。データ除染と quality reweighting の効果が大きい
  - **128K で RULER 27B 66.0 の限界**: 32K では 91.1 と SOTA だが、128K に外挿すると 66.0 に劣化。**Qwen3-VL の 1M YaRN で 99.5%** に大きく劣る。Gemma 3 の「**5:1 attention + sliding window 1024**」は KV キャッシュ削減には効くが、**真の超長文脈処理には届かない**
  - **MoE 不採用**: InternVL 3.5 (241B-A28B) / Qwen3-VL (235B-A22B) と対照的に、Gemma 3 は **全モデル Dense**。**「consumer-grade hardware 動作**」という焦点と、**「MoE は推論メモリ要求が複雑」** という Google の判断を反映
  - **訓練データ非公開**: InternVL 3 (OpenGVLab/InternVL-Data) のような完全公開ではない。**Apache 2.0 風ライセンス**だが訓練データは非公開で、これは Google の商用 Gemini との分業を反映
  - **arXiv 2503.19786 と発表時期**: 2025 年 3 月公開で、Qwen2.5-VL (2025 Feb) のすぐ後、InternVL 3 (2025 Apr) の直前という重要なタイミング。MMMU 70% 突破ラッシュの一員
  - **次の ingest 候補**: PaliGemma / PaliGemma 2 原典（Google 系の別系列、Gemma 3 と分業）/ Gemma 1 / Gemma 2 原典（Gemma 系列の前史）/ Gemini 1.5 / 2.0 Technical Report（商用 Gemini の対比、ただし公式論文形式があれば）/ BOND / WARM / WARP（事後学習 RL 手法、Google）/ Chameleon 原典（QK-norm の元祖、Meta）/ Olmo 2（QK-norm 採用）/ Phi-3 / Phi-4（Microsoft の軽量 MLLM 対比）/ TinyVLA / MobileVLM（軽量 VLM 競合）

## [2026-05-31] ingest | DeepSeek-OCR: Contexts Optical Compression

- 取り込み: `raw/papers/DeepSeek-OCR_ Contexts Optical Compression.md`（103 行、Web Clipper 形式 markdown、arXiv:2510.18234、Haoran Wei + Yaofeng Sun + Yukun Li、DeepSeek-AI、2025 Oct）
- **特記事項**: 元の markdown が **Web Clipper の抽出不完全（Abstract + References のみ、本文なし）** だったため、ar5iv（https://ar5iv.labs.arxiv.org/html/2510.18234）から WebFetch ツールで本文を取得して翻訳。これは **wiki ingest プロセスの新パターン**：Web Clipper markdown が不完全な場合、ar5iv 等から補完するワークフロー
- 作成:
  - [[sources/deepseek-ocr]] — DeepSeek-OCR の要約（LLM 中心視点での VLM 再定義、SAM+16×ConvNet+CLIP の DeepEncoder、DeepSeek-3B-MoE-A570M デコーダ、6 解像度モード、Fox/OmniDocBench SOTA、Memory Forgetting Mechanism）
  - [[translations/deepseek-ocr]] — Abstract + §1-10 の本文翻訳（ar5iv ベース、図 2-4 + 全主要テーブル含む）
  - [[entities/deepseek-ocr]] — DeepSeek-OCR のエンティティ詳細（DeepEncoder 構成、デコーダ仕様、6 モード、訓練パイプライン、全ベンチ結果、限界、系譜）
  - `raw/assets/deepseek-ocr/`: fig1.png（既存 VLM エンコーダ比較）、fig2.png（DeepEncoder アーキテクチャ）、fig3.png（5 解像度モード文書例）、fig4.png（大画像、結果比較）、fig5.png（大画像、結果比較）の 5 画像を ar5iv からダウンロード
- 更新:
  - [[index]] — Sources（DeepSeek-OCR を Gemma 3 の前に配置）、Translations（同上）、Entities Models（同上）の 3 セクションに追加
  - [[entities/sam]] — 「後継・派生モデル」表に DeepSeek-OCR を追加（SAM-base を Visual Perception 成分として活用、segmentation 用途ではなく文書 OCR の効率的視覚エンコーダに再利用される新応用例）
  - [[entities/clip]] — 「産業応用への影響」セクションに DeepSeek-OCR を追加（CLIP-large を Visual Knowledge 成分として組み込む新応用パターン）
  - [[concepts/foundation-model]] — A. 画像-テキスト WSL 系セクションに DeepSeek-OCR を Gemma 3 の前に追加（OCR 特化 MLLM の代表）
  - [[questions/vit-dynamic-resolution-evolution]] — Phase 2/3 タイル分割路線の表に DeepSeek-OCR を追加（「視覚トークン数最小化」という第 3 路線）、まとめ図解、関連ページに追加
  - [[overview]] — 最新更新ラインに DeepSeek-OCR を追加、VLM/MLLM 系統の **「OCR / 文書理解特化系」**を新規カテゴリとして追加、ingest 済みリストに追加
  - [[log]]
- メモ:
  - **DeepSeek-AI 初の wiki 登録の意義**: DeepSeek-V3 / R1 は Qwen3-VL や Gemma 3 / Qwen3.5-Omni の比較対象として頻出していたが、**DeepSeek-* モデルの独立 ingest は初めて**。これにより wiki の中国系 AI ラボ系譜が **Alibaba (Qwen 系) + Shanghai AI Lab (InternVL 系) + DeepSeek-AI** の **3 系譜**に拡張
  - **「視覚はテキストの効率的圧縮媒体」という新パラダイム**: Qwen / InternVL / Gemma 系の「視覚を視覚として扱う」哲学とは正反対の主張。**「LLM 中心視点で VLM を再定義」** という大胆な観点が新規性
  - **SAM + CLIP を直列に組み合わせる新しい視覚エンコーダ設計**: SAM（窓注意で高解像度処理）→ 2 層 ConvNet（16× 圧縮）→ CLIP（大域注意で意味抽出）の 3 段アーキテクチャ。**SAM と CLIP の wiki 既存ページから DeepSeek-OCR への接続**を確立
  - **視覚トークン数の劇的削減を SOTA で達成**: MinerU2.0 比 8.5× 効率、GOT-OCR2.0 比 2.56× 効率という効率性。**Qwen2.5-VL-72B（3949 トークン、0.214）/ InternVL3-78B（6790 トークン、0.218）も 800 トークン未満で凌駕**は衝撃的
  - **同著者 Haoran Wei の系譜**: Vary → Vary-tiny → Focus Anywhere → Slow Perception → GOT-OCR2.0 → DeepSeek-OCR の **6 作系列**。GOT-OCR2.0 の正統な後継として位置付けられる
  - **Memory Forgetting Mechanism の革新性**: 「人間の記憶減衰と視覚知覚劣化の並行性」を活用した文脈圧縮提案。**古い対話を低解像度画像にレンダリング → 自然な忘却曲線を模倣**。これは **LLM の長文脈問題への新解決策**として、Qwen3-VL の 1M YaRN 外挿や Gemma 3 の 5:1 attention とは全く異なるアプローチ
  - **DeepSeek-VL2 系列との分業不明確**: DeepSeek-AI は別途 DeepSeek-VL2（[^40] で参照）という汎用 VLM 系列を持つ。**DeepSeek-OCR は OCR 特化分枝**で、汎用 VLM 系列とは別系統と推測されるが、両系列の関係は論文中で明示されていない
  - **DeepSeek-3B-MoE-A570M の超軽量 MoE**: 推論時 570M 活性は **Qwen3-VL-30B-A3B (3B 活性) や InternVL 3.5 241B-A28B (28B 活性) より遥かに小さい**。「コンパクトな MoE で OCR 専門化」という戦略
  - **訓練データの公開度**: モデル重みは GitHub で公開だが、訓練データの完全な公開はなし（InternVL 3 の OpenGVLab/InternVL-Data 完全公開とは対照的）。**訓練データの内訳は記述されているが、データセット自体は非公開**
  - **OCR 1.0 と OCR 2.0 の分類**: OCR 1.0 = 自然テキスト文書 (30M+ ページ、シーン OCR)、OCR 2.0 = 構造化要素 (チャート → HTML 表、化学式 → SMILES、平面幾何 → 座標)。**「OCR 2.0」という用語は GOT-OCR2.0 (Wei et al., 2024c) で初出**、Haoran Wei の独自分類
  - **本文不在の Web Clipper 抽出問題**: 今回の markdown は Abstract + References のみで本文なし。これは Obsidian Web Clipper の抽出ルール（ar5iv のページ構造）に依存する問題。**今後の Web Clipper 抽出失敗時のフォールバック手順**として、ar5iv URL 直接 fetch を確立できた
  - **arXiv 2510.18234 と発表時期**: 2025 年 10 月公開で、Qwen3-VL (2025 Nov) の直前、Gemma 3 (2025 Mar) より後。MMLM 競争の中で **OCR 特化路線** が独立カテゴリとして確立される時期
  - **次の ingest 候補**: GOT-OCR2.0 原典 (Wei et al., 2024c, arXiv:2409.01704)（DeepSeek-OCR の直接の前任、本論文で主要ベースライン）/ Vary 原典 (ECCV 2024)（DeepSeek-OCR 著者の前作）/ DeepSeek-VL2 原典 (arXiv:2412.10302)（DeepSeek-AI の汎用 VLM 系列）/ DeepSeek-V3 / R1 (DeepSeek-AI LLM 系列、Qwen3-VL/Gemma 3 で頻繁に比較対象として参照)/ MinerU2.0 (主要 OCR パイプライン)/ dots.ocr / OCRFlux / SmolDocling / Nougat / olmocr（OCR 競合）/ OmniDocBench 原典（評価ベンチマーク）/ Slow Perception (Wei et al., 2024d, 幾何図形の段階的知覚)

## [2026-05-31] ingest | SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis

- 取り込み: `raw/papers/SDXL_ Improving Latent Diffusion Models for High-Resolution Image Synthesis.md`（588 行、Web Clipper 形式 markdown、arXiv:2307.01952、Stability AI Applied Research、2023 Jul）
- **特記事項**: **ユーザー指示で Appendix を含む**翻訳（Acknowledgements [Appendix A] と References は除外）
- 作成:
  - [[sources/sdxl]] — SDXL の要約（5 主要改善、Stable Diffusion 系統での位置、生成モデル wiki 初の代表として詳細）
  - [[translations/sdxl]] — Abstract + §1-3 + Appendix B-J の本文 + 付録翻訳（References と Acknowledgements 除外）
  - [[entities/sdxl]] — SDXL のエンティティ詳細（base + refiner + VAE 仕様、3 条件付け、訓練レシピ、全結果、系譜）
  - `raw/assets/sdxl/`: fig1 (SDXL vs SD)、fig2 (size dist)、fig3 (size cond)、fig4 (comparison)、fig5 (crop cond)、fig6 (refiner)、fig12 (FID vs CLIP) の 7 画像を ar5iv からダウンロード
- 更新:
  - [[index]] — Sources / Translations / Entities Models の 3 セクションに追加（DeepSeek-OCR の前に配置）
  - [[concepts/diffusion-model]] — Stable Diffusion 系統の系譜詳細を追加（SDXL を中心に LDM → SDXL → SDXL Turbo → Stable Cascade → SD3 → FLUX.1）
  - [[entities/clip]] — 「産業応用への影響」セクションに SDXL を追加（CLIP ViT-L + OpenCLIP ViT-bigG の二重エンコーダの最大実用例）
  - [[overview]] — 最新更新ライン更新、Foundation Model 3 大系統に **「拡散モデル系（生成）」**を新規追加、ingest 済みリストに SDXL を追加
  - [[log]]
- メモ:
  - **wiki 初の拡散モデル系統 ingest の意義**: これまで Qwen/InternVL/Gemma 等の MLLM、SAM/DINO 等の認識系、CLIP/SigLIP 等の対比学習を ingest してきたが、**生成モデル系の代表が完全に欠落**していた。SDXL の ingest により [[concepts/diffusion-model]] が **理論ページから実装例ページへの接続**を獲得。Computer Vision wiki の **「認識 + 生成」の両輪**が整う
  - **「Open Foundation Model の代表」としての SDXL**: 商用 Midjourney v5.1 を **54.9%** で凌駕した時点（2023 年 7 月）で、オープンソース text-to-image が初めて商用フロンティアと互角になった歴史的瞬間。**Stability AI の代表的成果**として位置付けられる
  - **3 つの新規条件付け技術の業界的影響**: size/crop/multi-aspect conditioning は **すべて後続の拡散モデル（SD3、FLUX、Imagen 2、Stable Cascade、Würstchen 等）で標準化**。**フーリエ特徴で timestep に加算するパターン**が拡散モデル分野の de facto 標準に
  - **39% データ廃棄問題の解決**: size-conditioning がデータ廃棄問題を **完全に解消**したという発見は、後の拡散モデル訓練手法に決定的影響。**「最小解像度しきい値」というアドホック手法を学術的に終わらせた**
  - **Concept bleeding 問題の発見と CLIP テキスト・エンコーダの限界**: SDXL 段階で **CLIP の単一トークン圧縮と対比損失が属性バインディングを失敗させる**ことを明確に指摘。後の **SD3 / FLUX で T5-XXL 大規模 LLM エンコーダへ移行**する強い動機となった
  - **FID/CLIP 古典指標の信頼性問題**: Pick-a-pic（Kirstain et al., 2023）の **「COCO ゼロショット FID は視覚美学と負の相関」**発見を SDXL が **追加的に裏付け**。これは生成モデル評価の **大きなパラダイム転換**を促進した
  - **Refinement Model パラダイム**: SDEdit ベースの 2 段階生成（base → refiner）は、その後の eDiff-I、IDM-VTON、AYS、Hunyuan-DiT 等で発展。**「単一モデルでは品質と多様性のトレードオフが厳しい」**という洞察に基づく汎用的なアプローチ
  - **Rombach lab の系譜**: LDM 原論文（Rombach et al., 2022）→ SDXL（2023）→ SD3（2024）と続き、その後 Rombach ら主要メンバーが Stability AI を離脱して **Black Forest Labs を設立、FLUX.1 を発表**。SDXL は **拡散モデル史における最も影響力のある実装の 1 つ**
  - **Stable Diffusion 系統と DiT 系統の対立**: SDXL は **UNet ベース**の最終形で、その後の SD3 / FLUX は **DiT (Diffusion Transformer)** に移行。SDXL 論文の §3 で UViT/DiT を試したが「即時利益なし」と判断したが、後の世代で見直された。**UNet vs DiT の対立**は拡散モデル史の重要な軸
  - **Stability AI の歴史的意義**: 商用閉源モデルが支配する分野で **完全オープンで品質を競う**という姿勢は、後の Mistral、DeepSeek、Black Forest Labs 等の **オープン基盤モデル運動**の先駆け
  - **CLIP の主要応用例として記録**: CLIP ViT-L + OpenCLIP ViT-bigG の二重エンコーダは、CLIP/OpenCLIP の **最大規模の実用例**。後のすべての Stable Diffusion 系統で踏襲
  - **arXiv 2307.01952 と発表時期**: 2023 年 7 月公開で、これは ChatGPT 公開（2022 Nov）から約 8 ヶ月後、Midjourney v5 リリース（2023 春）の数ヶ月後。**生成 AI 元年（2022-2023）の最後の主要オープンモデル**としての位置付け
  - **40 アスペクト比表は重要な参考資料**: Appendix I の 40 アスペクト比表（0.25 〜 4.0、~1024² 面積）は後の拡散モデル訓練でしばしば参照される実用的詳細
  - **次の ingest 候補**: Latent Diffusion (Rombach et al., 2022) 原論文（SDXL の直接の前任）/ DDPM 原典 (Ho et al., 2020、拡散モデルの祖)/ DDIM (Song et al., 2020、決定論的サンプラー)/ Classifier-free Guidance (Ho & Salimans, 2022)/ EDM (Karras et al., 2022、連続時間 DM)/ DiT (Peebles & Xie, 2022、Diffusion Transformer)/ SDEdit (Meng et al., 2021、image-to-image)/ Imagen (Saharia et al., 2022)/ DALL-E 2 (Ramesh et al., 2022)/ Stable Diffusion 3 (2024、DiT + T5)/ FLUX.1 (Black Forest Labs, 2024、Rombach ら独立)/ Pick-a-pic (Kirstain et al., 2023、FID 信頼性問題)

## [2026-05-31] ingest | Foundational Models in Vision: A Survey and Outlook

- 取り込み: `raw/papers/Foundational Models Defining a New Era in Vision_ A Survey and Outlook.md`（Awais et al., MBZUAI ら, arXiv:2307.13721 / IEEE TPAMI、1215 行・274KB）
- 作成:
  - [[sources/foundational-models-vision-survey]] — 要約ページ（CV 基盤モデルの 4 軸分類体系を中心に整理、本 wiki 既存 entity との接続を明示）
  - [[translations/foundational-models-vision-survey]] — 本文翻訳（Abstract + §1-9、References 除外、Appendix なし）、図 1/2/3/4/5/13/14/15 を `<figure>` で埋め込み
  - `raw/assets/foundational-models-vision-survey/` フォルダ + 8 図（fig1-evolution / fig2-architectures / fig3-taxonomy / fig4-clip-variants / fig5-clip-errors / fig13-sam / fig14-sam-medical / fig15-palm-e）
- 更新:
  - [[concepts/foundation-model]] — **「サーベイによる体系的分類（Awais et al., TPAMI 2024）」** セクションを追加、4 軸（テキスト・プロンプト型 / 視覚プロンプト型 / 異種モダリティ型 / 身体性型）+ 4 アーキテクチャ・スタイル（Dual-Encoder / Fusion / Encoder-Decoder / Adapter LLM）を整理、既存 entity 群（CLIP/SigLIP/GLIP/Grounding-DINO/Qwen-VL 系/InternVL 系/Gemma 3/DeepSeek-OCR/SAM 系/PE/Qwen3.5-Omni）を分類体系に位置付け、sources frontmatter に追加
  - [[index]] — Sources / Translations セクションに各 1 行追加
  - [[overview]] — 最新更新ラインに追加（updated: 2026-05-31）
  - [[log]]
- メモ:
  - **wiki 初のサーベイ論文 ingest**: これまでは個別モデル論文（CLIP / SAM / DINOv2 / Qwen-VL 系等）の ingest が中心で、**「複数モデルを横断的に分類する 2 次文献」の ingest は本件が初めて**。これにより wiki に「個別 entity」と「分類体系」の 2 層構造が確立
  - **survey が提供する「鳥瞰図」の価値**: 本 wiki は 2025 年 1 月時点で 50+ の entity ページ（CLIP / SAM / DINO 系 / Qwen-VL 系 / InternVL 系 / Gemma 3 / DeepSeek-OCR 等）を持つが、**「これらが共通の体系のどこに位置するか」を一覧で示すページが欠落**していた。survey の 4 軸（テキスト/視覚/異種/身体性）+ 4 スタイル（Dual-Enc/Fusion/Enc-Dec/Adapter LLM）の分類体系を [[concepts/foundation-model]] に統合したことで、初学者は「wiki 全体の地図」を得られるようになった
  - **survey の発表時期（2023 年中期）と内容のズレ**: 本 survey は arXiv 2023 Jul の論文で、wiki に登録された GPT-4V / Qwen-VL 後継（Qwen2-VL/2.5-VL/3-VL）/ InternVL 後継（1.5/2.5/3/3.5）/ Gemma 3 / DeepSeek-OCR / SAM 2/3 などは含まれない。**しかし survey の分類体系は普遍的で、これら新世代モデルも 4 軸のいずれかに正確に位置付けられる**（実際に source ページと concepts/foundation-model でその位置付けを明示）
  - **survey の scope 限定（VL のみ）**: 本 survey は明示的に純粋画像 SSL（DINOv2/v3/MAE 等）と画像生成系（拡散モデル等）を対象外と宣言。本 wiki はこれらを **独立した別軸として並列管理**（DINO 系統 / MAE / iBOT / BYOL / 拡散モデル系 = SDXL）。**両軸を統合する「視覚基盤モデル」の包括的 survey は依然欠落しており、本 wiki がその役割を担う構造**
  - **Adapter LLM パターンの事実上の標準化**: survey が定義した 4 アーキテクチャ・スタイルのうち、2024-2025 で **Adapter LLM** がほぼすべての対話型 MLLM の標準となった。BLIP-2 の Q-Former、Qwen-VL の Position-aware VL Adapter、InternVL の MLP projector、Gemma 3 の SigLIP + P&S + average pooling などはすべて **Adapter LLM カテゴリ内の異なる実装**として理解できる。この観察は wiki 全体の MLLM 理解に新しい統一視点を与える
  - **PVS / PCS の theoretical lineage**: survey は SAM の PVS（[[concepts/promptable-segmentation]]）を「軸 2: 視覚プロンプト型」の中核として扱う。SAM 3 の PCS（[[concepts/promptable-concept-segmentation]]）は survey 後の発展だが、軸 2 の自然な拡張として位置付けられる
  - **対話型 VLM が軸 4（身体性型）へ侵食する流れ**: Qwen2.5-VL の **ScreenSpot Pro 43.6**、Qwen3-VL の **OSWorld 38.1**、InternVL 3.5 の **WindowsAgentArena** は「軸 1（対話型 VLM）が軸 4（仮想環境エージェント）に侵食している現代の流れ」を象徴。survey 執筆時（2023 Jul）には未予測の発展軸で、本 wiki でその流れを継続観察する重要性が確認された
  - **次の ingest 候補との関連**: 本 survey は CV 基盤モデルの「全体地図」を与えるため、今後 ingest する論文（拡散モデル系統 / Latent Diffusion / DDPM / DiT、または新たな MLLM や SSL モデル）も、本 survey の分類体系に位置付けることで、wiki 全体の整合性が保てる

## [2026-06-03] ingest | RoFormer: Enhanced Transformer with Rotary Position Embedding

- 取り込み: `raw/papers/RoFormer_ Enhanced Transformer with Rotary Position Embedding.md`（Su et al., Zhuiyi Technology Co., Ltd., 深圳, 2021 Apr arXiv:2104.09864 / Neurocomputing 2024、801 行・56KB、Web Clipper 形式）
- 作成:
  - [[sources/roformer]] — 要約ページ（RoPE の核心アイデア、3 つの性質、4 つの実験、CV 既存ページとの接続を整理）
  - [[translations/roformer]] — 本文翻訳（Abstract + §1-5、References 除外、Appendix なし）、図 1-3 を `<figure>` で埋め込み、表 1-5 を含む
  - `raw/assets/roformer/` フォルダ + 3 図（fig1-rope-illustration / fig2-long-term-decay / fig3-training-loss）
- 更新:
  - [[concepts/rotary-position-embeddings]] — **frontmatter の sources に `[[sources/roformer]]` を追加**（既存の DINOv3/SAM 2/Qwen2-VL/Qwen2.5-VL/Qwen3-VL に加え原典として）、**一言まとめで `[[sources/roformer\|"RoFormer", 2021]]` への明示的リンク**、**関連ページ冒頭に「RoPE の原典論文」として最優先表示**
  - [[index]] — Sources（RoFormer = RoPE 原典）/ Translations セクション、略称表に **RoFormer / PLM / GLUE / MRPC など / CAIL2019-SCM / WoBERT / NEZHA / Performer / long-term decay / Abel transformation** を追加（既存の RoPE 行も `[[sources/roformer]]` 併記に更新）
  - [[overview]] — 最新更新ライン（updated: 2026-06-03）
  - [[log]]
- メモ:
  - **CV wiki に NLP 論文を初 ingest した意義**: 本 wiki はこれまで CV / VL モデル中心だったが、RoFormer は NLP 論文でありながら **CV / Vision Transformer / MLLM の「任意解像度対応」の理論的基礎**として最重要原典。DINOv3 の axial RoPE、SAM 2 の memory 2D-RoPE、Qwen2-VL 系の M-RoPE はすべて本論文の RoPE を出発点とする。**「NLP で生まれたが CV で開花した技法」の代表例**として、wiki に原典が欠落していたことを補完
  - **RoFormer 概念ページとの関係明確化**: 本 wiki には既に [[concepts/rotary-position-embeddings]] が存在し、RoPE の概念・CV 採用モデル・派生（NTK-aware/YaRN/LongRoPE/M-RoPE/Interleaved MRoPE）まで包括的に整理されていた。今回の ingest は **「概念ページの原典を遡って ingest した」初の事例**。概念ページ → 原典への双方向リンクを確立
  - **論文の歴史的経路**: 2021 年 4 月 arXiv 初版時点では英語圏で大きく注目されなかった。著者 Jianlin Su は中国 Zhuiyi Technology の研究者で、ブログ記事 **"Transformer 升级之路"** で先に発想を公開していたため、論文の経路が NLP 主流の発表チャネルと違った。**LLaMA（Meta, 2023）が採用したことで一気に英語圏で標準化**され、現在に至る
  - **GLUE では全勝でなかった点を誠実に記録**: SST-2 で -2.8、QNLI で -2.5、MNLI で -4.4/-3.6 と BERT に負けている。**「位置エンコーディングだけ替えれば常に勝てる」わけではない**。後の世代で他の改良と組み合わせて初めて全面的優位に。論文時点の比較は単純で、3/6 タスクで超えるという「部分勝利」だった
  - **本論文がスコープ外とした話題（後の研究で重要に）**:
    - **長期外挿**（context length extrapolation）: 訓練時の系列長を大きく超えると性能劣化する問題は本論文では触れず → 後の **NTK-aware RoPE / YaRN / LongRoPE** で解決
    - **2D / 3D 拡張**: 画像・動画への拡張は完全に後続研究の貢献（axial RoPE, M-RoPE, Interleaved MRoPE）
    - **線形 attention との両立は §3.3 で短く言及のみ**: 後の Performer 系研究が本格化させた
  - **3 つの実験的観察の重要性**:
    - **WMT 翻訳で BLEU +0.2 のみ** — Translation のような短文タスクでの改善は限定的
    - **MLM 事前学習で収束加速** — 著者自身が「なぜ速いか説明できない」と認める観察
    - **CAIL2019-SCM 中国語法律文書 1024 で WoBERT +1.5%** — **長文タスクで本領発揮**、これが後の長文 LLM 採用の根拠に
  - **数式的明快さ**: §3.2 の「2D 複素数導出 → d 次元への一般化」と §3.4 の「アーベル変換による長期減衰証明」は、後の派生研究すべての基盤となる定式化。**特に `θ_i = 10000^{-2i/d}` という選択が Vaswani 2017 の正弦波エンコーディングと同じ系列を使う**点は、「正弦波を加算から乗算に変換した」という解釈を可能にする
  - **本 wiki にとっての位置付け**: [[concepts/foundation-model]] の survey 分類で「Adapter LLM」スタイルが事実上の標準となった現代において、**RoPE は Adapter LLM のすべての実装が共有する基盤技術**。今後 ingest する任意の MLLM 論文も、その position encoding は RoPE 系（M-RoPE / Interleaved MRoPE / YaRN 等）と読み解ける
  - **次の ingest 候補**（RoPE 系の拡張・改良）: NTK-aware RoPE（系列長外挿）/ YaRN（NTK-aware の改良、Mistral/Qwen 採用）/ LongRoPE（200K+ 系列対応）/ Attention is All You Need（Vaswani et al., 2017、Transformer 原典で正弦波 encoding の元）/ T5 paper（相対位置バイアスの代表）/ DeBERTa（disentangled attention）/ Performer（線形 attention、§4.4 で評価対象）/ Linear Transformers（Katharopoulos et al., 2020）

## [2026-06-03] ingest | An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

- 取り込み: `raw/papers/An Image is Worth 16x16 Words_ Transformers for Image Recognition at Scale.md`（Dosovitskiy et al., Google Research Brain Team, 2020 Oct arXiv:2010.11929 / ICLR 2021、615 行・67KB、Web Clipper 形式）
- 作成:
  - [[sources/vision-transformer]] — 要約ページ（ViT アーキテクチャの 4 ステップ、B/L/H 命名規約、3 つのスケーリング知見、CV 既存ページとの広範な接続を整理）
  - [[translations/vision-transformer]] — 本文翻訳（Abstract + §1-5、References と Appendix A-D は除外、CLAUDE.md §4 デフォルトに従う）、図 1-7 を `<figure>` で埋め込み、表 1-2 を含む
  - `raw/assets/vision-transformer/` フォルダ + 6 図（fig1-architecture / fig2-vtab / fig3-data-requirements / fig5-scaling / fig6-attention-examples / fig7-internal-representations）
- 更新:
  - [[concepts/vision-transformer]] — **frontmatter sources に `[[sources/vision-transformer]]` を原典として追加**（既存の DINO 原典の前に配置）、関連ページ冒頭に「ViT 原典」として最優先表示、一言まとめに **「ImageNet 88.55% / VTAB 77.63% / 1/4 以下計算」** の重要な数値を追加
  - [[index]] — Sources / Translations セクション、略称表に **ViT / ViT-B/L/H / MSA / JFT-300M / BiT / Noisy Student / VTAB / iGPT / TPUv3-core-days / attention distance / inductive bias / GELU / [CLS] token / pre-norm** を追加（15 項目）
  - [[overview]] — 最新更新ライン（updated: 2026-06-03）
  - [[log]]
- メモ:
  - **本 wiki の「根幹原典」を補完した意義**: ViT は CV における基盤モデル時代の出発点で、本 wiki に登録済みのほぼすべての視覚モデル（DINO/MAE/iBOT/DINOv2/DINOv3/CLIP/SigLIP/SAM/PE/Qwen-VL 系/InternVL 系/Gemma 3/DeepSeek-OCR）が ViT を視覚バックボーンとして採用。**「概念ページが頻繁に参照しているのに原典がない」という致命的欠落を、RoFormer に続いて補完**。これで wiki の主要な「概念→原典」リンクが完成
  - **RoFormer ingest と同じパターン**: 概念ページ（concepts/vision-transformer.md、2026-05-24 作成）が既に存在し、後から原典を遡って ingest した **「概念ページの原典補完」第 2 例**。RoFormer (RoPE) に続く事例で、wiki としての設計パターンが確立
  - **論文の発表時期と影響**: 2020 年 10 月 arXiv 初版、2021 年 ICLR。発表時点では「Transformer が画像にスケールするか」を確認する研究として注目されたが、**その後 5 年以内に CV のすべてのサブ分野（分類・検出・セグメンテーション・VLM・MLLM・生成）が ViT に移行**するという、CV 史上最大級のパラダイム・シフトを引き起こした
  - **3 つのスケーリング知見の歴史的重要性**:
    - **(1) データ・サイズが帰納バイアスの差を埋める** — これが後の DINOv2 (142M), DINOv3 (1.689B), CLIP (400M), SigLIP, JFT-3B などすべての「Web スケール訓練」の根拠
    - **(2) 小データで ViT-L < ViT-B、大データで逆転** — 「大きなモデル = 必ず良い」ではなく「大きなモデル × 大きなデータ」が必要というメッセージ。後の Chinchilla scaling law（NLP）と平行的
    - **(3) 試した最大スケールで未飽和** — CV の「スケール限界はまだ見えていない」を示し、ViT-7B（DINOv3）/ ViT-22B（Google）/ EVA-1B/2B などの巨大化を動機づけた
  - **JFT-300M 問題**: ViT の最良結果は JFT-300M（Google 内部非公開）に依存。**この再現性問題が**:
    - OpenCLIP / LAION-400M / LAION-5B の登場（オープン代替）
    - DINOv2 の LVD-142M / DINOv3 の LVD-1689M の自前キュレーション
    - **オープン基盤モデル運動全体**の根拠の一つ
  - **位置埋め込み議論の含意**: 論文 §3.1 と Appendix D.4 で「1D ≈ 2D ≈ 相対位置 ≈ no PE（IN-21k 事前学習時）」と結論。**これが「位置埋め込みの選択は重要でない」という当時のコンセンサスを作った**が、後に間違いと判明：
    - **任意解像度対応**の観点では学習可能 1D は致命的に弱い（事前学習時の系列長を超えると補間が必要）
    - **DINOv3 の axial RoPE** や **Qwen2-VL の M-RoPE** が登場し、**[[sources/roformer|RoPE]] が事実上の標準**に
    - 本論文の比較は ImageNet 系評価に限定され、任意解像度・任意系列長への汎化を評価していなかった
  - **「シンプルさ」の哲学**: 論文の中心メッセージは **「Transformer に画像固有の修正を加えない」**。これは美しい設計思想だが、後の研究で:
    - **Hiera/Swin**（階層型）が密予測タスクで必要に
    - **Window Attention**（Qwen2.5-VL）が高解像度の計算量問題を解決
    - **DeiT/CaiT**（強い拡張・蒸留）が小データでの ViT 訓練を可能に
    - **MAE/iBOT/DINO**（SSL）が JFT 依存を解消
    - これらすべてが「シンプル ViT を実用化する」ための補完研究として理解できる
  - **Hybrid モデルの教訓**: 論文 §4.4 で **「Hybrid（CNN + ViT）は小モデルでわずかに勝つが大モデルで差消失」**。これは「**最終的には純粋 ViT が勝つ**」という強いメッセージで、CNN の歴史的役割を終わらせた重要な実験結果
  - **[CLS] トークンと 1D 位置埋め込みの今**: 本論文で確立された [CLS] + 1D 位置埋め込みは、**現在の MLLM 系では大部分が変更されている**:
    - Qwen-VL 系: 2D-RoPE（任意解像度）+ Merger（量を制御）
    - DINOv3: axial RoPE + register tokens（[CLS] の代替・補完）
    - Gemma 3: SigLIP 400M + average pooling 256 トークン圧縮
    - **本論文の「初期設計」を尊重しつつ、ほぼすべての MLLM が独自改良している**
  - **§4.6 自己教師あり予備実験の重要性**: わずか 1 段落だが、**「BERT 風マスク・パッチ予測で ViT-B/16 が ImageNet 79.9% を達成」** という観察が、後の **MAE（He et al., 2021/2022, ImageNet only で 87.8%）** や **iBOT / BEiT / SimMIM** に直結。**論文の予備実験が分野全体の新しいパラダイム（Masked Image Modeling）を生んだ**事例
  - **次の ingest 候補**（ViT 系の自然な続き）:
    - **DeiT**（Touvron et al., 2021）— 強い拡張 + 蒸留で ImageNet のみで ViT 訓練を可能に
    - **Swin Transformer**（Liu et al., 2021）— 階層型 ViT、密予測の標準
    - **Attention is All You Need**（Vaswani et al., 2017）— Transformer 原典、まだ wiki にない致命的欠落
    - **BERT**（Devlin et al., 2018）— ViT が直接モデルとした事前学習パラダイムの原典
    - **iGPT**（Chen et al., 2020）— ViT 直前の「画像 × Transformer」の先行研究
    - **CoCa / BEiT / SimMIM** — ViT 後の主要 SSL 系統

## [2026-06-03] query | 大規模事前学習の系列別整理

- ユーザ質問: 「近年の大規模データによる自己教師あり学習には CLIP の系列と DINO の系列があると思っています。これはあっていますか？他にも系列がありますか？それぞれの系列の発展（アーキテクチャの進化、データの大規模化、革新的な工夫）を整理してください」
- 作成: [[questions/large-scale-pretraining-series]] — 大規模事前学習の **5 系列**（①WSL 対比型 / ②自己蒸留 SSL / ③MIM / ④古典対比 SSL（pre-ViT）/ ⑤ハイブリッド）を整理
- 更新: [[index]] / [[overview]]
- メモ:
  - **ユーザの認識への診断**: 「CLIP 系 + DINO 系」の 2 大主流の把握は実用上正しいが、**用語的には WSL（CLIP は弱教師あり）vs SSL（DINO は自己教師あり）の区別**あり。冒頭で明示
  - **欠落していた 3 系列を新規追加**: ③MIM 系（MAE/BEiT、純粋系列は終焉でハイブリッドへ吸収）/ ④古典対比 SSL 系（SimCLR/MoCo/BYOL/SwAV、pre-ViT、現代 SSL の理論的基礎）/ ⑤ハイブリッド系（iBOT/DINOv2/v3、純粋 SSL の現代主流）
  - **MLLM の Vision encoder への直接的回答**: ユーザが「LLM の Vision encoder」を CLIP 系の枝として挙げた点は概ね正しい。Qwen-VL 系（DFN/SigLIP-2 継承）/ InternVL 系（CLIP-ViT-L 初期化 + 継続事前学習）/ Gemma 3（SigLIP 400M 共有）/ DeepSeek-OCR（SAM + CLIP 直列）— **すべて CLIP/SigLIP 系の継承で独立系列ではない**
  - **2024-2025 の系列融合論点を整理**: SigLIP 2（WSL が SILC 自己蒸留 + TIPS マスク予測を借用）、PE（WSL が alignment tuning で中間層活用 + SAM 2.1 蒸留）、DINOv3（SSL が CLIP 由来 RoPE 採用）。「系列の境界が曖昧化しつつあるが監督信号の出所での区別は根本的」と結論
  - **次の ingest 候補を系列ごとに提示**: 系列①（ALIGN / MetaCLIP / DFN / AIMv2）/ 系列③（BEiT / SimMIM / V-JEPA）/ 系列④（MoCo / SwAV）。**wiki に未登録の重要原典が WSL/MIM/古典対比に偏っていることが可視化された**
  - **ハブ的 question ページの価値**: [[questions/vit-dynamic-resolution-evolution]] に続く 2 つ目の question ページ。**個別 entity ページや concepts は揃っているが「系列横断の俯瞰」は [[concepts/foundation-model]] のセクションが唯一だった**ところに、独立 question として体系的整理を加えた

## [2026-06-03] schema-update | frontmatter の wikilink を Obsidian 有効な YAML に修正

- 問題: Obsidian で wiki/concepts, entities, questions の各ページが「**無効なプロパティ**」と表示されていた（添付スクリーンショット）
- 原因: CLAUDE.md スキーマで指定していた `related: [[link1]], [[link2]]` という形式が **YAML 文法的に無効**（`[` で始まる値は YAML フローシーケンスとして解析されるが、`[[link]]` はネストした空リスト + 不明識別子と解釈されエラー）
- 修正:
  - **`wiki/` 配下の 166 ファイル × 計 238 行**を一括修正（合計 610 wikilink を保持）
  - **複数値フィールド** `related` / `sources` / `sources_used` → インライン YAML リスト + 引用符付き文字列: `related: ["[[a]]", "[[b]]"]`
  - **単数値フィールド** `translation` / `source_page` → スカラー引用符付き文字列: `translation: "[[t]]"`
  - **wikilink を含まないフィールド**（`aliases`, `tags`, `authors`, `year` 等）は無変更（従来通り `tags: [vit, transformer]` 等で有効）
  - **CLAUDE.md §2 Frontmatter 規約**を更新: 5 つのテンプレート（sources/translations/concepts/entities/questions）すべてで正しい引用符付き形式に修正、**冒頭に「Obsidian の YAML プロパティ規約」セクションを追加**して今後の ingest で誤らないように
  - **CLAUDE.md §4 翻訳ファイルのテンプレート**の `source_page` も修正
- 検証:
  - 修正後に「不正な unquoted `[[...]]`」が 0 ファイル
  - 全 wikilink（frontmatter 内 610 個）が保持されている
  - 5 タイプ全てで spot check OK（concepts/sources/translations/entities/questions）
- メモ:
  - **影響範囲**: 本文中の `[[wikilink]]` には影響なし（マークダウン本文の wikilink は Obsidian が正しく処理する。YAML フロントマターのみの問題）
  - **YAML フロー記法の罠**: `aliases: [ViT, Vision Transformer]` は **valid YAML**（フローシーケンス、内容は文字列 "ViT" と "Vision Transformer"）。一方 `related: [[a]], [[b]]` は **invalid YAML**（先頭の `[` でフローシーケンス開始だがその中に `[a]` ネストして閉じ忘れ + コンマ区切りの後続要素が不正）
  - **過去の ingest すべてに影響**: 2026 年 5 月以降に作成した 166 ファイルすべてに同問題が存在。今回一括修正で解決
  - **今後の予防**: CLAUDE.md スキーマ修正により、次の ingest からは初出で正しい形式で frontmatter を書ける。新規ページ作成時に「無効なプロパティ」表示の再発がないことを確認するのが望ましい

## [2026-06-03] ingest | YaRN: Efficient Context Window Extension of Large Language Models

- 取り込み: `raw/papers/YaRN_ Efficient Context Window Extension of Large Language Models.md`（Peng et al., Nous Research + EleutherAI + University of Geneva, 2023 arXiv:2309.00071 / ICLR 2024、587 行・60KB、Web Clipper 形式）
- ユーザー指示: **「appendix 含めて ingest」**
- 作成:
  - [[sources/yarn]] — 要約ページ（PI → NTK-aware → NTK-by-parts → Dynamic NTK → YaRN の 5 世代を整理、Qwen3-VL 1M YaRN 外挿との接続を明記）
  - [[translations/yarn]] — 本文 + Appendix 翻訳（**Abstract + §1-6 + Appendix A.1, A.2, B.1, B.2, B.3, B.4 含む**、References のみ除外）、図 1-6 を `<figure>` で埋め込み、表 1-6 を含む
  - `raw/assets/yarn/` フォルダ + 6 図（fig1 sliding-window-perplexity / fig2 mscale-vs-ppl overall / fig3 mscale-vs-ppl segment / fig4 mscale-vs-ppl argmin / fig5 dynamic-scaling / fig6 mistral-perplexity）
- 更新:
  - [[concepts/rotary-position-embeddings]] — **「派生」セクションを大幅拡張**: PI → NTK-aware → Dynamic NTK / NTK-by-parts → YaRN → LongRoPE の 5 世代系譜を ASCII 図と数式定義で整理、frontmatter sources に `[[sources/yarn]]` を追加、関連ページに YaRN 行追加
  - [[index]] — Sources / Translations セクション、略称表に **YaRN / PI / NTK-aware / NTK-by-parts / Dynamic NTK / Dynamic-YaRN / LongRoPE / scale factor s / wavelength $\lambda_d$ / ramp function $\gamma(r)$ / Passkey Retrieval / PG19 / Proof-pile / GovReport / Flash Attention 2 / bloc97 / emozilla** の 17 項目追加
  - [[overview]] — 最新更新ライン（updated: 2026-06-03）
  - [[log]]
- メモ:
  - **wiki の RoPE 系列を完成させる ingest**: RoFormer（RoPE 原典）→ ViT 原典 → **YaRN（RoPE 拡張の決定版）** の 3 連で、wiki の「位置エンコーディング × 任意系列長」の理論的基礎が揃った。**Qwen3-VL の 1M YaRN 外挿に対する明確な原典参照が成立**
  - **Reddit / GitHub コミュニティ駆動研究の代表事例**: NTK-aware（bloc97 Reddit 投稿）/ Dynamic NTK（emozilla Reddit 投稿）/ NTK-by-parts（bloc97 GitHub PR）はいずれもピアレビュー前のコミュニティ発見だった。本論文はこれらを **「事後的に学術形式に整える」役割**。著者 Bowen Peng 自身が Reddit `/u/bloc97` で NTK-aware の発見者。**Nous Research + EleutherAI + Reddit コミュニティの生態系**を象徴する論文
  - **本論文の真の貢献**: 「新規アイデアの提案」より「**4 つの先行手法（PI, NTK-aware, Dynamic NTK, NTK-by-parts）の整理 + 1 つの新規要素（温度スケーリング）の統合**」。**YaRN そのものは増分的改良**で、整理と統合に重きがある
  - **5 世代の系譜の知識的価値**: wiki に **PI / NTK-aware / NTK-by-parts / Dynamic NTK / YaRN の数式定義と歴史**が整理されたことで、今後の Qwen / Mistral / Llama 系の MLLM ingest でこれらの用語を **概念ページに飛ばすだけで済む** ようになった。MLLM 各モデルが「どの拡張手法を採用しているか」を明確に区別可能に
  - **CV / MLLM での意義**:
    - **[[entities/qwen3-vl|Qwen3-VL]]**: 256K ネイティブ + 1M YaRN 外挿、Needle 256K 100% / 1M 99.5% は **本論文の YaRN $s=32$ の MLLM スケール延長**
    - **[[entities/qwen2-vl|Qwen2-VL]]**: 学習 16K → 推論 80K 外挿は YaRN 系技法に依拠
    - **DINOv3 の RoPE-box jittering**（[[sources/dinov3]]）: 訓練時に座標範囲をランダムスケール、**YaRN の Dynamic Scaling と発想的に近い**
    - 純粋 Vision モデルでの YaRN 直接適用論文はまだ稀（Vision では DINOv3 の axial RoPE のように **RoPE 自体の 2D 化** が先行）
  - **訓練効率の革命**: 「事前学習データの 0.1% 未満、400 ステップ」で 4k → 128k 拡張は **以前不可能だった効率**。後続の長文脈手法（LongRoPE 2024、Qwen の各 long-context モード）が **YaRN の効率水準を当然視**するようになった
  - **「Effective context size」と perplexity の乖離**（B.2）: Code Llama 13B は 100k 超で perplexity 増加するが、Passkey 検索 128k で 99.4%。**perplexity 単独では長文脈評価指標として不十分**という重要な観察。これが後の Needle-in-a-Haystack / RULER / LongBench 等の長文脈ベンチマーク登場を促した
  - **未解決の論点**: (1) 温度 $\sqrt{1/t}=0.1\ln(s)+1$ の理論的説明欠如、(2) Llama 2 への一般化が「普遍性」かどうか、(3) MLLM 文脈での YaRN ベストプラクティスは依然エンジニアリング知識、(4) Vision での YaRN 直接適用論文の少なさ
  - **次の ingest 候補**（RoPE 系完成後の自然な発展）: LongRoPE（Ding et al., 2024、200K 超）/ Position Interpolation 原典（Chen et al., 2023、arXiv:2306.15595）/ ALiBi（Press et al., 2022、ICLR、長系列外挿の代替パラダイム）/ Code Llama（Roziere et al., 2023、NTK-aware の最初の本格採用）/ Together LLongMA-2（PI の代表応用）/ V-JEPA（Vision での長系列・動画 SSL）

## [2026-06-03] ingest | Data Filtering Networks（DFN）

- 取り込み: `raw/papers/Data Filtering Networks.md`（Fang et al., Apple + University of Washington, 2023 Sept arXiv:2309.17425 / ICLR 2024、418 行・51KB、Web Clipper 形式）
- ユーザー指示: **「appendix 含めて ingest」**
- 作成:
  - [[sources/dfn]] — 要約ページ（手法・3 つの非自明発見・CLIP 系列での位置付け・MLLM での採用を整理）
  - [[entities/dfn]] — エンティティページ（DFN モデル仕様、DFN-2B/DFN-5B データセット詳細、4 つの公開チェックポイント、CLIP 系列との比較）
  - [[translations/dfn]] — 本文 + Appendix A-G 翻訳（**Abstract + §1-5 + Appendix A-G 含む**、References のみ除外）、図 1-8 を `<figure>` で埋め込み、表 1-10 を含む
  - `raw/assets/dfn/` フォルダ + 8 図（fig1 compute-scaling / fig2 pipeline / fig3 filtering-vs-IN / fig4 data-quality / fig5 robustness / fig6 avg-fig3 / fig7 avg-fig4 / fig8 log-scale）
- 更新:
  - [[concepts/weakly-supervised-pretraining]] — **「A. 画像-テキスト対比学習（CLIP 系）」セクションに DFN を追加**、PE の後・Qwen 系の前に配置、3 つの鍵発見と Qwen2-VL/2.5-VL での採用を明記、frontmatter sources に `[[sources/dfn]]` 追加
  - [[index]] — Sources / Translations / Entities Models セクション、略称表に **DFN / DFN-2B / DFN-5B / HQITP-350M/135M / DataComp / DataComp-1B / CommonPool / filter dataset / induced dataset / induced model / OAI-Init / M3AE / AXlearn / CC12M / CC3M / SS15M** の 15 項目追加
  - [[overview]] — 最新更新ライン（updated: 2026-06-03）
  - [[log]]
- メモ:
  - **wiki の MLLM 視覚エンコーダ系譜が完成**: Qwen2-VL/2.5-VL の ViT 初期化に使われている DFN が ingest 済みになり、**「視覚エンコーダの出自」を Qwen-VL 全世代について明示的に追跡可能**に：
    - Qwen-VL（2023.08）: OpenCLIP ViT-bigG
    - **Qwen2-VL（2024.09）: DFN ← 今回**
    - **Qwen2.5-VL（2025.02）: DFN + 社内データ ← 今回**
    - Qwen3-VL（2025.11）: SigLIP-2 から継続学習
  - **Apple 系の wiki 第 1 弾**: 本 wiki に Apple 系モデルがまだなかった。**DFN ingest で Apple Hugging Face（apple/DFN5B-CLIP-ViT-H-14-378 等）と Apple のデータ・キュレーション哲学を取り込んだ**。次の Apple 系候補は AIMv2（autoregressive image modeling）、HQITP（人間検証キャプション）、MM1（Apple MLLM）
  - **3 つの非自明発見が現代的に重要**:
    - **「フィルタ性能 ≠ ImageNet 性能」**: 別領域でも応用可能な深い洞察。「下流タスクで強いモデル」が「データ選別に強いモデル」と一致しないことを実証。これは **MLLM の vision encoder 評価でも「分類ベンチで強い ≠ MLLM で強い」** という観察に通じる
    - **「データ品質が決定的」**: 図 4 の即座崩壊カーブは劇的。**Web スケール訓練でデータ品質を維持することが量より重要**という主張
    - **「CLIP フィルタ > バイナリ分類器」**: マルチモーダル整列が単純な分類より柔軟なフィルタ基準を提供するという原理的観察
  - **DataComp との関係**: 並行発表の DataComp（Gadre et al. 2023）と相補的。**DataComp = ベンチマーク + ベースライン・データ (DC-1B)、DFN = ベンチマーク上で SOTA を取る手法**。両者が「データ中心 AI」の代表ペア
  - **MetaCLIP との対比**: MetaCLIP（Xu et al. 2024、ICLR）はデータ公開で相補的、DFN は手法公開。**MetaCLIP は CLIP 訓練データの選別基準を初公開、DFN はその選別をニューラルネットで学習可能にする**。両者が異なる軸で CLIP データを民主化
  - **SigLIP 2 との比較**: SigLIP 2 (2025、85.0% IN) は **5 系統融合**（対比 + 自己蒸留 + MIM + decoder + 蒸留）で DFN-5B（84.4%）を上回るが、**SigLIP 2 は損失関数 + 補助タスクの革新、DFN はデータ・キュレーションの革新**。**直交する改善軸で、原理的には組み合わせ可能**
  - **HQITP-350M の謎**: 「人間検証済みキャプション 357M」という強力なデータの出自・キュレーション基準が論文中で詳細未公開。**「Apple 内部資産」とのみ記述**。再現性の最大の障害だが、§4.2 で公開データのみでも競争的 DFN 構築可能と示すことで一部緩和
  - **DFN-5B の 30B 非 DataComp 画像**: 出自完全非公開。**完全再現が困難**だが、公開された DFN-2B（DataComp 12.8B から選別）は完全再現可能
  - **MLLM 視覚エンコーダの選択理由が明確化**: Qwen2-VL/2.5-VL が DFN を選んだ理由は **「DFN-2B の HQITP-FT 効果で多言語 OCR / Document / Chart のキュレーション品質が高い」** と推測される。実際 Qwen2-VL は多言語 OCR で GPT-4o を 7/8 言語で凌駕しており、DFN の HQITP のような高品質キャプションへのファインチューンが効いている可能性
  - **データ中心 AI の意義**: 本論文の最大の貢献は「**データセット設計はモデル設計と同じツールを使える**」という主張。これは：
    - DataComp（ベンチマーク化）と相補的
    - MetaCLIP（データ公開）と相補的
    - SigLIP 2 / PE のキュレーション哲学に影響
    - 「Data-Centric AI」運動の代表事例
  - **未解決の論点**: (1) 「フィルタ性能 ≠ IN 性能」の理論的説明欠如、(2) HQITP-350M の中身（人間検証基準）が不透明、(3) DFN-5B の 30B 非 DataComp 画像の出自不明、(4) 再帰的改善（DFN-5B モデルを新しい DFN として使う）は未検証、(5) 動画・3D・専門ドメインでの DFN は未検証
  - **次の ingest 候補**（CLIP 系列の wiki 補完）: AIMv2（Apple 2024、autoregressive image modeling、DFN の対抗パラダイム）/ MetaCLIP（Meta ICLR 2024、データ公開）/ EVA-CLIP（BAAI 2023、ViT-G/14 5B）/ DataComp 原典（Gadre et al. 2023、ベンチマーク詳細）/ MM1（Apple 2024、Apple 系の MLLM、DFN を使った）/ Florence-2（Microsoft 2024、統合 vision foundation）
