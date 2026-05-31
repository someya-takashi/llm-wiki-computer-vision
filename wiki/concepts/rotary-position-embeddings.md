---
type: concept
aliases: [RoPE, Rotary Position Embeddings, 回転位置埋め込み, Rotary Positional Encoding]
tags: [architecture, transformer, position-encoding]
related: [[vision-transformer]]
sources: [[sources/dinov3]], [[sources/sam-2]]
updated: 2026-05-24
---

# Rotary Position Embeddings（RoPE, 回転位置埋め込み）

## 一言で

**Transformer の attention の中で、トークン間の相対位置を「回転行列を用いた変換」として表現する位置埋め込み手法**。Su ら（"RoFormer", 2021）が NLP のために提案し、現在は LLaMA・Mistral・PaLM・Qwen など主要 LLM の事実上の標準となっている。Computer Vision でも DINOv3（[[entities/dinov3]]）や RoPE-ViT などで採用が進む。**任意の解像度／系列長で fine-tuning なしに使える**のが視覚での最大の利点。

---

## なぜ位置埋め込みが必要か

Transformer の self-attention は**順序を全く考慮しない**。トークン列を入力するとき「どれが 1 番目で、どれが 2 番目か」をモデルに教える機構が別途必要になる。これが位置埋め込み（positional embedding）。

CV では ViT が画像を 16×16 のパッチに切って系列化するので、「どのパッチが画像のどこにあるか」を伝える必要がある。

---

## 既存の位置埋め込みとの比較

### 1. Learnable absolute positional embedding（学習可能絶対位置埋め込み）

オリジナル ViT [Dosovitskiy 2020] が採用。

- 各位置に対して学習可能なベクトルを 1 つ用意
- パッチ埋め込みに**足し合わせる**
- 長所: シンプル、学習が容易
- 短所: **解像度が固定**される。学習時 224×224（14×14 = 196 パッチ）で訓練すると、推論時に 512×512 などを入れたければ補間が必要

### 2. Sinusoidal positional encoding（正弦波位置エンコーディング）

オリジナル Transformer（Vaswani 2017）が採用。

- 各位置を sin / cos 関数の値で表現
- 長所: 学習パラメータ不要、ある程度長さに汎化
- 短所: パッチ埋め込みに足すだけなので、attention で相対位置を直接活用できない

### 3. Relative positional embedding（相対位置埋め込み）

Shaw ら（2018）, Swin Transformer 等が採用。

- attention の計算式中に「クエリとキーの相対位置」によるバイアスを直接挿入
- 長所: 相対距離をモデルが直接認識できる
- 短所: 実装が複雑、attention の計算コストが上がる

### 4. RoPE（本記事の主題）

- attention の Q (query) と K (key) ベクトルに**回転行列**を掛けて位置情報を埋め込む
- 内積 $Q \cdot K$ を取ると、回転の差分が残るため**相対位置が自動的に attention に現れる**
- パラメータ追加なし、計算コスト微増、解像度に汎化可能

---

## RoPE のアイデア（概要）

### NLP（1 次元）の場合

位置 $m$ にあるトークンの Q ベクトル $q_m$ を、位置に応じた角度 $\theta$ だけ回転させる：

$$
q_m' = R_m \cdot q_m
$$

ここで $R_m$ は 2 次元の回転行列（ベクトルを 2 次元ずつに区切り、それぞれを回転）。同様に位置 $n$ の K ベクトル $k_n$ に $R_n$ を適用。

すると内積は：

$$
q_m' \cdot k_n' = (R_m q_m) \cdot (R_n k_n) = q_m \cdot (R_{n-m} k_n)
$$

つまり**相対位置 $n - m$ だけが attention の値に影響する**。これが「相対位置を陽に扱う」という性質。

> **補足: 直観的な例え** — 各トークンが「位置に応じた角度で時計の針が回っている」と考える。2 つのトークンの「時計の針の差分」だけが内積に影響する。絶対的な時刻はキャンセルされ、相対的な時間差だけが効く。

### 2 次元（画像）への拡張: axial RoPE

画像はパッチが 2 次元グリッド上に並ぶので、X 軸と Y 軸の位置をそれぞれ別個に RoPE で扱う。

- 特徴量の半分を X 軸の回転に、もう半分を Y 軸の回転に割り当てる（"axial"）
- パッチが $(x, y)$ にあるなら、対応する Q と K の前半に $R_x$、後半に $R_y$ を適用

DINOv3 はこの **axial RoPE** を採用。

---

## DINOv3 における RoPE の使い方

[[entities/dinov3]] と [[sources/dinov3]] §3.2 から：

- **正規化座標**: パッチ位置を $[-1, 1]$ の範囲に正規化（画像サイズに依存しない）
- **RoPE-box jittering**: 訓練中、座標範囲 $[-1, 1]$ をランダムに $[-s, s]$（$s \in [0.5, 2]$）にスケール。これで様々なスケール・アスペクト比に頑健になる
- **解像度汎化**: 256² で訓練したモデルが推論時 **4096² でも安定に動く**

> **補足: なぜ RoPE が「あらゆる解像度」で動くか** — RoPE は「位置 $m$ に対して角度 $m\theta$ 回転」する操作。$m$ は連続値として扱えるので、訓練時に見たことがない位置（パッチ数）でも、回転角を計算するだけで対応できる。学習可能位置埋め込みは「位置 1, 2, ..., 196 用のベクトル」を訓練したきり拡張できないのと対照的。

---

## RoPE の長所と短所

### 長所

1. **相対位置を attention で直接扱える**: 学習効率が良い
2. **解像度・系列長への汎化**: 補間不要で任意の入力長に対応
3. **パラメータ追加ゼロ**: モデルサイズに影響しない
4. **計算コスト微増**: Q/K ベクトルへの行列適用だけ
5. **NLP/CV/音声で同じ実装が使える**: モダリティ非依存

### 短所

1. **絶対位置を扱いたいケースには弱い**: 「画像の左上か右下か」を明示的に区別したい場合
2. **head dimension が偶数である必要**: 2 次元回転を適用するため
3. **直観的理解が難しい**: 学習可能位置埋め込みより数学的
4. **古典的な位置への外挿能力には限界**: 訓練時の最大長を大きく超えると性能低下しうる（NLP 文脈、解像度面では vision でも長期的な検証が必要）

---

## 採用モデル

### NLP

- **LLaMA / LLaMA 2 / LLaMA 3**（Meta）
- **Mistral / Mixtral**
- **PaLM**（Google）
- **GPT-NeoX**
- **Qwen** / **Yi** / **DeepSeek** / **Falcon**
- 事実上の標準

### CV

- **RoPE-ViT**（Heo et al., 2024）: ViT への axial RoPE 適用
- **DINOv3**（2025, [[entities/dinov3]]）: axial RoPE + box jittering
- **SAM 2**（Meta, 2024, [[entities/sam-2]] / [[sources/sam-2]]）: **memory attention で 2D-RoPE を使用**。画像エンコーダ（Hiera）では RPB を全削除し絶対位置埋め込みのみ、memory cross-attention のみ 2D-RoPE。「memory への spatial 一貫性が必要だが画像エンコーダはシンプルに保ちたい」設計判断
- **EVA-02**: 一部派生で採用
- **InternVL**: マルチモーダルでも採用増加中

---

## 派生

- **NTK-aware RoPE**: 長系列への外挿性を改善（LLaMA で広く使われる）
- **YaRN**: NTK-aware の改良
- **LongRoPE**: 200K トークン以上の超長系列対応
- **2D RoPE / Mixed RoPE**: vision/video 向けの 2D・3D 拡張
- **M-RoPE（Multimodal RoPE）**: [[entities/qwen2-vl|Qwen2-VL]] が導入した、回転位置埋め込みを **temporal / height / width の 3 成分**に分解する MLLM 専用版。テキストは 1D-RoPE 等価、画像は temporal 固定 + 空間 2 軸で位置 ID、動画は temporal 増分 + 空間 2 軸。**画像と動画の位置 ID 値を小さく抑えて長文脈外挿に有利** という副次効果が重要。**学習 16K → 推論 80K トークンまで頑健**という Qwen2-VL-72B の長さ外挿はこれが決定的に効いている。後続の Qwen2.5-VL / Qwen3-VL や多くの MLLM が踏襲
- **MRoPE Aligned to Absolute Time**: [[entities/qwen2-5-vl|Qwen2.5-VL]] が導入した M-RoPE の改良版。Qwen2-VL の M-RoPE は **temporal ID をフレーム番号に結びつけていた**ため、30 fps 動画と 5 fps 動画で「同じ 10 秒」が異なる temporal ID 列で表現されていた。Qwen2.5-VL は **temporal ID 間の間隔を絶対時間（秒数）に揃える**ことで、異なる FPS の動画にわたって一貫した時間整合を学習できる。追加のテキスト・タイムスタンプ注入や追加ヘッドを必要としない（Qwen-Agent や他社 MLLM が採用するハック的アプローチを回避）。**Charades-STA mIoU 50.9（GPT-4o 35.7 を +15.2 圧倒）** と LVBench / MLVU SOTA の決定打。後続の MLLM で動画グラウンディングを扱う場合の標準アプローチに
- **Interleaved MRoPE**: [[entities/qwen3-vl|Qwen3-VL]] が導入した M-RoPE のさらなる改良。Qwen2-VL / Qwen2.5-VL の MRoPE は **temporal/height/width を埋め込み次元の「塊」に分割**して別々の回転周波数を割り当てていたため、**周波数スペクトル不均衡**を生み、長動画理解性能を劣化させていた（Huang et al., 2025）。Interleaved MRoPE は t / h / w 成分を埋め込み次元にわたって**均一に交互配置**することで、**各時空間軸が低周波帯と高周波帯の双方で均一に表現**されることを保証。動画の長距離位置モデリングを著しく改善。さらに、Qwen3-VL は **MRoPE absolute time も捨てて `<3.0 seconds>` のような明示的テキスト・タイムスタンプ・トークン**に切り替え、長動画での temporal ID 肥大化問題を解消（後続節「テキスト・ベース時間整合」参照）

---

## 関連ページ

- [[sources/dinov3]] — vision で axial RoPE を本格採用した論文
- [[sources/sam-2]] — memory attention で 2D-RoPE を採用、画像エンコーダ側は絶対位置埋め込みのみという設計選択
- [[sources/qwen2-vl]] — M-RoPE（temporal/height/width 3 成分）を導入し、ViT 側も 2D-RoPE で任意解像度対応にした論文
- [[sources/qwen2-5-vl]] — M-RoPE を絶対時間に整合させた論文（Qwen2-VL のフレーム番号紐付けの問題を解消、Charades-STA mIoU 50.9 SOTA）
- [[sources/qwen3-vl]] — Interleaved MRoPE（t/h/w を埋め込み次元に交互配置）+ テキスト・ベース時間整合（MRoPE absolute time から `<3.0 seconds>` 明示的タイムスタンプ・トークンへ転換）を導入
- [[entities/dinov3]] — RoPE-box jittering で解像度頑健性を強化したモデル
- [[entities/sam-2]] — 動画ストリーミング処理の memory に 2D-RoPE
- [[entities/qwen2-vl]] — M-RoPE を採用した MLLM、学習 16K → 推論 80K 外挿
- [[entities/qwen2-5-vl]] — MRoPE Aligned to Absolute Time を採用、Charades-STA で時間グラウンディング SOTA
- [[entities/qwen3-vl]] — Interleaved MRoPE + テキスト・ベース時間整合を採用、Charades-STA 64.8 SOTA、256K ネイティブ / 1M YaRN 外挿
- [[concepts/vision-transformer]] — RoPE が改善する対象アーキテクチャ
