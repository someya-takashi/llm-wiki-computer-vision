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
