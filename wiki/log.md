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
