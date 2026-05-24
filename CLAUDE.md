# Computer Vision LLM Wiki — スキーマ

このリポジトリは Andrej Karpathy が提唱した「LLM Wiki」パターンに基づく、Computer Vision（CV）領域のパーソナル・ナレッジベースです。LLM（あなた）が wiki を**読み・書き・更新する側**、ユーザーは**情報源のキュレーションと質問**を担当します。ユーザーは wiki を直接編集することはほぼありません。

このファイルは「あなた（LLM）」のための運用ルール書です。すべての ingest / query / lint 作業はこの schema に従って実行してください。

---

## 1. ディレクトリ構成

```
Computer Vision/
├── CLAUDE.md                ← このファイル（スキーマ）
├── raw/                     ← 原典（immutable, LLM は読むだけ）
│   ├── papers/              ← 論文 PDF（arXiv 等）
│   ├── articles/            ← ブログ記事・Webクリップ等の markdown
│   ├── images/              ← 記事に紐づかない単独画像
│   └── assets/              ← 記事内画像のローカル保存先
└── wiki/                    ← LLM が完全所有する markdown 群
    ├── index.md             ← カタログ（全ページの一覧）
    ├── log.md               ← 時系列ログ（append-only）
    ├── overview.md          ← CV 全体の総括ページ（随時更新）
    ├── sources/             ← 原典 1 件につき 1 ページの要約・解説
    ├── translations/        ← 原典の全文翻訳（要約せず一文ずつ正確に）
    ├── concepts/            ← 概念・手法・アーキテクチャの解説ページ
    ├── entities/            ← モデル・データセット・人物・組織のページ
    └── questions/           ← query で得た成果物（比較表・分析等）を保存
```

### 命名規約

- ファイル名は `kebab-case.md`（例: `vision-transformer.md`）
- 原典の wiki ページ（sources/translations）は元ファイル名にあわせる（例: `raw/papers/2020-vit.pdf` → `wiki/sources/2020-vit.md`, `wiki/translations/2020-vit.md`）
- 概念ページは英語の正式名称（略称ではなくフルネーム）をスラグにする
- 略称（ViT, CNN, IoU 等）は専用ページを作らず、対応する正式名称ページ（vision-transformer, convolutional-neural-network, intersection-over-union）にリダイレクトする位置づけで `index.md` に併記する

### 言語ポリシー

- **wiki 内の解説文（sources / concepts / entities / questions / overview / index / log）は日本語**で書く
- **translations のみ、原典が英語であれば日本語訳**を作成する（原典が日本語であれば翻訳は作成しない）
- 固有名詞・術語は無理に和訳せず、初出時に「Vision Transformer（ViT, 画像識別用 Transformer）」のように原語＋略称＋短い注釈を添える

### リンク規約

- 内部リンクは Obsidian の `[[wikilink]]` 記法を使う（`[[vision-transformer]]` のように slug を直接書く）
- まだ存在しないページへのリンク（dangling link）も許容する。`lint` 時に未作成ページとして検出する
- 原典への参照は `[[sources/2020-vit]]` のようにディレクトリ込みで書く

---

## 2. Frontmatter 規約

すべての wiki ページに YAML frontmatter を付与する。Obsidian Dataview で集計できるようにする。

### sources/*.md

```yaml
---
type: source
source_path: raw/papers/2020-vit.pdf
source_kind: paper            # paper | article | blog | video | podcast
title: An Image is Worth 16x16 Words
authors: [Alexey Dosovitskiy, ...]
year: 2020
venue: ICLR 2021
ingested: 2026-05-24
tags: [vit, transformer, image-classification]
translation: [[translations/2020-vit]]
---
```

### translations/*.md

```yaml
---
type: translation
source_path: raw/papers/2020-vit.pdf
source_page: [[sources/2020-vit]]
original_language: en
translated_to: ja
translated_at: 2026-05-24
---
```

### concepts/*.md

```yaml
---
type: concept
aliases: [ViT, Vision Transformer]
tags: [architecture, transformer]
related: [[attention-mechanism]], [[convolutional-neural-network]]
sources: [[sources/2020-vit]], [[sources/2021-deit]]
updated: 2026-05-24
---
```

### entities/*.md

```yaml
---
type: entity
entity_kind: model            # model | dataset | person | organization | benchmark
aliases: []
related: []
sources: []
updated: 2026-05-24
---
```

### questions/*.md

```yaml
---
type: question
asked: 2026-05-24
question: "ViT と CNN は ImageNet でどちらが強いか？"
sources_used: [[sources/2020-vit]], [[sources/2015-resnet]]
---
```

---

## 3. オペレーション

### 3.1 Ingest（原典の取り込み）

ユーザーが `raw/` 配下に新しいファイルを置き「これを ingest して」と指示したときの標準フロー：

1. **読解**：原典を読み、主要な主張・手法・結果・前提・限界を把握する。画像リンクが埋め込まれている場合は、画像 URL も保持し、必要に応じて画像も参照する（後述「画像の扱い」を参照）。
2. **対話**：ユーザーに重要な発見・疑問点・既存 wiki との関係（既知概念との接続、矛盾、新概念の発生）を簡潔に伝え、強調すべき観点をすり合わせる。
3. **翻訳ファイルの作成**（**論文・記事の場合は必須**）：`wiki/translations/<slug>.md` を作成し、原典の本文を**一文ずつ正確に翻訳**する。詳細は §4 を参照。
4. **要約ページの作成**：`wiki/sources/<slug>.md` を作成する。詳細は §5 を参照。**ここは機械的な要約にしない**。略称の定義、難概念の補足、初学者でも読める平易な説明を心がける（§6 参照）。
5. **概念・エンティティページの追加 / 更新**：原典で重要な役割を果たす概念・モデル・データセット等について、`wiki/concepts/` `wiki/entities/` に新規ページを作るか既存ページを更新する。**ここも初学者向けの丁寧な解説を心がける**。
6. **クロスリファレンスの更新**：関連する既存ページの「関連項目」「参考文献」セクション、frontmatter の `sources` / `related` を更新する。
7. **overview.md の更新**：必要に応じて CV 全体の総括に変化を反映する（新潮流・パラダイムシフト等）。
8. **index.md の更新**：新規作成したすべてのページをカテゴリ別に追記する。
9. **log.md への追記**：`## [YYYY-MM-DD] ingest | <タイトル>` のフォーマットで、何を取り込み・どのページを作成/更新したかを箇条書きで記録する。

1 件の ingest で 5〜15 ページが触られるのが普通。基本は **1 件ずつ**取り込み、ユーザーと対話しながら進める。

### 3.2 Query（質問）

ユーザーが wiki に対して質問してきたときのフロー：

1. **index.md を読む**：まず index で関連しそうなページを特定する。
2. **必要なページを読む**：drill-in し、複数ページを統合して回答する。原典に立ち戻る必要があれば `raw/` を直接読んでよい。
3. **回答する**：ユーザーの質問に対して、引用元ページを `[[wikilink]]` 形式で明示しながら回答する。**初学者にも分かるように略称展開・概念補足を入れる**（§6）。
4. **成果物の wiki 化を提案**：回答が比較表・分析・新しい接続の発見を含む場合、`wiki/questions/<slug>.md` として保存することをユーザーに提案する。ユーザーが了承したら作成し、index.md と log.md を更新する。

出力形式は質問に応じて選ぶ：markdown ページ、比較表、Marp スライド、matplotlib チャート、canvas 等。

### 3.3 Lint（健康診断）

ユーザーが「lint して」と指示したら、wiki 全体を点検する：

- **矛盾の検出**：同じ概念について異なる主張をしているページの組
- **古い記述**：新しい原典で更新されたはずなのに古い数値や主張が残っているページ
- **孤立ページ**：他のどのページからもリンクされていないページ
- **未作成ページ**：本文中で言及されているが対応ページが存在しない概念・エンティティ（dangling link）
- **欠落クロスリファレンス**：明らかに相互参照すべきページ同士でリンクが張られていない箇所
- **データギャップ**：Web 検索で埋められそうな情報の欠落

検出結果を一覧化してユーザーに提示し、優先度を相談する。修正は別途 ingest/query フローで行う。

---

## 4. 翻訳ファイル（wiki/translations）の仕様

### 何を翻訳するか

- 論文・記事の**本文**（abstract, introduction, related work, method, experiments, results, discussion, conclusion 等）
- **除外**：appendix（付録）、references（参考文献一覧）、acknowledgments（謝辞）

### 翻訳の方針

- **一文ずつ正確に翻訳する。要約・省略・意訳的圧縮はしない**。原典の論理構造をそのまま保持する。
- 原典の見出し階層をそのまま保持する（`## 1. Introduction` 等）
- 数式は LaTeX 記法のまま残す（`$...$` `$$...$$`）。数式中の変数定義は翻訳の地の文で日本語化してよい。
- 図表のキャプションも翻訳する。図表番号（Figure 1, Table 2 等）は保持する。
- 用語は初出時に「自己注意機構（self-attention）」のように **日本語訳（原語）** の形で書き、以降は文脈に応じて選択する。
- 原典に画像リンクが埋め込まれている場合、画像はそのまま該当位置に `![](パス)` で挿入する（§7 参照）。
- 翻訳ファイル内では、解釈や補足を加えない。**純粋に翻訳のみ**。補足説明は `wiki/sources/<slug>.md` の方に書く。

### 翻訳ファイルのテンプレート

```markdown
---
type: translation
source_path: raw/papers/2020-vit.pdf
source_page: [[sources/2020-vit]]
original_language: en
translated_to: ja
translated_at: 2026-05-24
---

# <原題（日本語訳）>

> 原題: <Original Title>
> 著者: ...
> 出典: ...

## Abstract（要旨）

<一文ずつの翻訳>

## 1. Introduction（はじめに）

<一文ずつの翻訳>

...
```

---

## 5. 要約ページ（wiki/sources）の仕様

要約ページは、原典を「読まなくても本質が掴め、かつ深く読みたくなったら原典に戻れる」状態を目指す。以下のセクションを基本構成とする（必要に応じて増減）：

```markdown
---
<frontmatter>
---

# <タイトル>

> 原典: [[translations/<slug>]] ・ `raw/papers/<slug>.pdf`
> 著者・年・会議: ...

## 一言まとめ

（1〜2 文で「何をした論文／記事か」）

## 背景と問題意識

（なぜこの研究／記事が出てきたのか。前提知識・先行研究の何が不足していたのか。**初学者でも理解できるよう、関連する概念を平易な言葉で補足する**）

## 提案手法 / 主張

（何をどう変えたのか。図解の引用があると望ましい）

## 実験結果と知見

（何が示されたのか。数値・比較対象も含める）

## 限界・批判的視点

（原典の限界、後の研究による反証、未解決問題など）

## 用語と略称

（本ページに出てきた略称をすべて展開・定義する。例: ViT = Vision Transformer、IoU = Intersection over Union）

## 関連ページ

- [[concepts/...]]
- [[sources/...]]
- [[entities/...]]
```

---

## 6. 「機械的なまとめ」にしない（最重要ルール）

`wiki/translations/` 以外のすべてのページ（sources / concepts / entities / questions / overview）で守ること：

1. **略称は必ず初出時に展開する**。`ViT` → `ViT（Vision Transformer, 画像を 16x16 パッチに分割して Transformer に通す画像分類モデル）` のように、**展開＋短い意味付け**をセットで書く。
2. **難概念は補足を入れる**。専門用語（例: inductive bias, locality, soft attention, ablation study）が出てきたら、その場で 1 文の補足説明を付ける。リンクで概念ページに飛べる場合でも、文脈で必要なら補足を残す。
3. **初学者の読者を想定する**。学部高学年〜修士 1 年程度の CV 初学者が読んで「何のことを言っているか分からない」段落を作らない。逆に、自明な内容を冗長に説明するのも避ける。
4. **原典の章立てをそのままコピーしない**。要約ページの構造は §5 のテンプレートに従い、原典の構造を「再解釈」した形にする。
5. **「研究の意義」を自分の言葉で説明する**。原典の Abstract をなぞるだけにしない。「なぜこの結果が CV 分野にとって重要なのか」を 1〜2 文加える。
6. **既存 wiki との接続を明示する**。「これは [[attention-mechanism]] を画像に持ち込んだ最初の本格的試みである」のように、既存知識と結びつける一文を必ず入れる。

逆に、`wiki/translations/` では上記ルールは **適用しない**。翻訳ファイルでは補足・解釈・接続づけは一切せず、原典に忠実に訳すことに専念する。

---

## 7. 画像の扱い

原典に画像が含まれる場合：

- **画像 URL がローカルに存在しない場合**：可能なら `raw/assets/<source-slug>/figN.png` のようなパスにダウンロードする（ユーザーが Obsidian Web Clipper + ダウンロード hotkey で自動化している場合あり）。ダウンロードできない場合は元 URL をそのまま使う。
- **要約ページ（sources）と翻訳ページ（translations）の両方で**、必要な箇所に下記「7.1 キャプション書式」に従って画像を引用する。
- **読解時の挙動**：LLM は markdown 中のインライン画像を 1 パスで読めないため、まず本文を読み、必要なら参照画像を別途 Read ツールで開いて文脈を補強する。
- 概念ページ（concepts）でもアーキテクチャ図などを引用してよい。引用元（どの source の図N か）を必ずキャプションに書く。

### 7.1 キャプション書式（Obsidian 対応）

Obsidian は Markdown 標準の alt テキストをキャプションとして表示しない。CSS スニペット不要で Reading view にそのままキャプションを出すために、**HTML の `<figure>` + `<figcaption>` で Markdown 画像記法を包む方式**を採用する。

```markdown
<figure>

![](../../raw/assets/<source-slug>/figN.png)

<figcaption>図N: ここにキャプション本文。原典のキャプションを翻訳した内容、または要約／概念ページでの文脈付き説明。</figcaption>
</figure>
```

ルール：
- `<figure>` の**直後と `</figure>` の直前に空行を入れる**こと。これがないと Obsidian は中の Markdown 画像記法を解釈せず HTML ブロックとして閉じてしまう。
- `<figcaption>` の**前にも空行を入れる**。
- 画像パスは Markdown 記法 `![](path)` を維持する（Obsidian のパス解決が効くため）。`<img src="">` は使わない。
- `<figcaption>` 内では **`$...$` などのインライン LaTeX を使わない**。確実に動かないので、Unicode 添字（`x₁`, `x²` 等）や簡易表記（`k-NN`, `224^2`）で代替する。
- キャプション本文は原典のキャプションに合わせ「図N: ...」または「表N: ...」で始める。要約／概念ページで再掲する場合は「図1（再掲）: ...」のように明示する。
- `alt` テキスト（`![](path)` の `[]` 内）は空でよい。内容は `<figcaption>` 側に書く。

表（Markdown テーブル）には `<figure>` を使わず、テーブル直上に `**表N**: ...` の太字キャプションを置くだけにする（原典スタイルの踏襲）。`<figure>` は図画像にのみ使う。

---

## 8. index.md と log.md の運用

### index.md

カテゴリ別の全ページカタログ。1 行 = 1 ページ。フォーマット：

```markdown
- [[<slug>]] — <一行の説明>
```

セクション：
- Overview
- Sources（さらに paper / article 等に分けてよい）
- Translations
- Concepts
- Entities（model / dataset / person / organization）
- Questions

ingest / query で新規ページを作るたびに必ず更新する。

### log.md

時系列の append-only ログ。

```markdown
## [YYYY-MM-DD] ingest | <タイトル>

- 取り込み: `raw/papers/2020-vit.pdf`
- 作成: [[sources/2020-vit]], [[translations/2020-vit]], [[concepts/vision-transformer]]
- 更新: [[concepts/attention-mechanism]], [[overview]], [[index]]
- メモ: ...
```

`grep "^## \[" log.md | tail -10` で直近の動きを追えるよう、必ずこのプレフィックス形式を守る。

---

## 9. ツール

- **Obsidian**：wiki の閲覧・グラフビュー確認。ユーザーが裏側で開いている。
- **Obsidian Web Clipper**：記事を markdown 化して `raw/articles/` に保存。
- **Marp**：スライド出力が必要な質問への回答に使う（任意）。
- **Dataview**：frontmatter ベースの動的集計（任意）。

検索は規模が小さいうちは index.md ベースで十分。ページ数が増えてきたら `qmd` 等の導入を検討する。

---

## 10. このスキーマ自体について

このスキーマはユーザーと共進化する。運用していく中で「このカテゴリが必要」「このルールは緩めたい」となったら、ユーザーに提案してこの CLAUDE.md 自体を更新する。スキーマ変更は log.md にも `## [YYYY-MM-DD] schema-update | <要点>` として記録する。
