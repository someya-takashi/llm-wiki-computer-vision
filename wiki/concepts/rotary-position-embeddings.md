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

---

## 関連ページ

- [[sources/dinov3]] — vision で axial RoPE を本格採用した論文
- [[sources/sam-2]] — memory attention で 2D-RoPE を採用、画像エンコーダ側は絶対位置埋め込みのみという設計選択
- [[entities/dinov3]] — RoPE-box jittering で解像度頑健性を強化したモデル
- [[entities/sam-2]] — 動画ストリーミング処理の memory に 2D-RoPE
- [[concepts/vision-transformer]] — RoPE が改善する対象アーキテクチャ
