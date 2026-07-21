# ROGII Public LB 5 点台を狙う独自モデル戦略

策定日: 2026-07-20

## 1. 位置付け

この文書は、公開上位 Notebook の fork、係数変更、公開 prediction の blend を主手段とせず、自分たちで学習・検証したモデルから Public LB `5.xxx` を狙うための研究計画である。

公開 Notebook は以下にのみ使用する。

- score と計算量の control
- feature、validation、失敗例の参考
- 独自モデルが本当に異なる誤差を持つかの最終比較

独自提出の主予測には、ravaghi、fleongg、pilkwang の学習済み prediction を入力しない。公開モデルを teacher にした distillation も、独自モデルの性能が確立するまでは行わない。

## 2. 難易度と前提

再現可能な公開 Notebook の frontier は約 6.96--7.10 である。Discussion には LB 6.577、CV/LB 5.x の報告があるが、完全な実装は公開されていない。

したがって、5 点台には保証されたレシピがない。次のどれかが必要になる。

- 公開 PF lineage が捉えていない新しい signal
- GR alignment の分岐曖昧性を扱う確率モデル
- 井戸間・地層面の空間構造を leak-free に利用するモデル
- prefix、typewell、horizontal GR、trajectory を同時に扱う sequence/path model
- tail wells の大誤差を減らす risk-aware training

単にモデルを大きくするだけでは達成できない。1D-CNN、Transformer、cross-attention は、GR の自己相似性による誤った branch への confidence jump を起こしやすい。モデル構造に path continuity と uncertainty を明示的に入れる必要がある。

## 3. 問題の再定義

坑井行 `i` の TVT は、既知の軌跡深度 `Z_i` と地層面レベル `U_i` に分解できる。

```text
TVT_i = U_i - Z_i
U_i   = TVT_i + Z_i
```

`Z_i` は test でも既知である。そのため、高周波な坑井形状を TVT model に学習させる必要はない。主に推定すべき対象は、well ごとに比較的低周波な `U` の drift、dip、branch である。

独自モデルは絶対 TVT を直接回帰せず、次の階層で構成する。

```text
regional surface prior
  + known-prefix state
  + GR/typewell alignment likelihood
  + constrained path posterior
  + small learned residual
```

最終評価が RMSE なので、二峰性が残る場合は最尤 branch ではなく posterior mean を基本出力とする。

## 4. 検証設計を最初に固定する

5 点台を狙う場合、モデルより先に validation を固定する。Public LB に合わせてから検証を作らない。

### 4.1 外側 fold

773 wells を単位とする固定 5-fold GroupKFold を使用する。同じ well の row を train と validation に分けない。

### 4.2 spatial holdout

X/Y を field block に分け、近隣 wells を同時に validation にする leave-spatial-block-out を併用する。

空間 model、formation imputer、normalizer、retrieval index は fold ごとに train wells だけで再構築する。validation well を query 時に除くだけでは不十分である。

### 4.3 typewell/region holdout

同一または非常に近い typewell、formation regime を group 化した holdout も診断用に持つ。近い well の暗記と、未知領域への汎化を分離する。

### 4.4 multi-cut simulation

各 train well に対して複数の pseudo cutoff を作り、異なる prefix 長から suffix を予測させる。

- actual-like cutoff
- prefix 25%、50%、75% 付近
- suffix 長を test 分布に合わせた cutoff
- GR 欠損率、Z span、坑井長で stratify

複数 cutoff はデータ拡張には使うが、外側 fold は well 単位のまま維持する。

### 4.5 評価指標

公式の pooled row RMSE に加え、次を必ず保存する。

- fold RMSE
- well median / p90 / max RMSE
- worst 5% / 10% SSE share
- mean bias RMSE
- slope/dip error
- suffix distance 別 RMSE
- spatial distance 別 RMSE
- branch ambiguity 別 RMSE
- seed variance
- paired well bootstrap CI

## 5. 独自モデルの全体構成

```text
                  typewell GR / geology
                           |
                           v
horizontal GR ---- learned emission/alignment model
      |                    |
      |                    v
known TVT prefix --> constrained K-best path lattice
      |                    |
      v                    v
regional surface prior -> posterior mean path
                           |
                           v
                  residual TCN / tree
                           |
                           v
                 physics projection on U
```

構成要素は単独で OOF 評価し、最終的に decorrelated なものだけを blend する。

## 6. Model A: regional stratigraphic surface prior

### 6.1 目的

GR だけでは解けない datum と低周波 dip を、近隣 wells と formation surface から推定する。

### 6.2 target

```text
U = TVT + Z
delta_U = U - U_at_last_known_prefix
```

### 6.3 入力

- X、Y、MD、Z、方位、傾斜
- last-known `U` と prefix の `dU/dMD`
- ANCC、ASTNU、ASTNL、EGFDU、EGFDL、BUDA
- formation thickness と相対 phase
- train wells から構築した local plane / residual field
- nearest-well distance と spatial uncertainty

### 6.4 候補

- CatBoost / LightGBM の delta-U model
- local polynomial surface + residual CatBoost
- sparse Gaussian Process または RBF field
- graph model は tree/spatial baseline が確立してから検討

空間 prior は単独で最終予測にせず、後段 path model の prior mean と uncertainty に使う。

## 7. Model B: learned GR emission model

### 7.1 目的

従来 PF の `horizontal GR - typewell GR` の二乗誤差を、局所形状を理解する学習済み similarity に置き換える。

### 7.2 学習データ

train well の真の TVT を使い、horizontal GR window と、正しい typewell TVT 周辺の GR window を positive pair にする。

negative pair は難易度を分ける。

- 近い offset の hard negative
- 自己相似ピークの hard negative
- 別 formation の negative
- 別 well/typewell の negative

### 7.3 モデル

最初は小型 Siamese 1D CNN または TCN とする。

- horizontal GR window encoder
- typewell GR/geology window encoder
- normalized embedding similarity
- offset classification または contrastive loss

大型 cross-attention は、単純 emission model が従来 NCC を上回ってから試す。

### 7.4 判定

row RMSE の前に alignment 診断を行う。

- true offset rank
- top-1 / top-k accuracy
- correct branch と false repeated branch の margin
- known prefix の held-out alignment error
- spatial/typewell holdout の calibration

従来の normalized correlation を安定して上回らなければ path decoder に入れない。

## 8. Model C: constrained probabilistic path lattice

このモデルを独自主経路の中心にする。

### 8.1 state

各 MD step で候補となる `U` または typewell TVT offset を離散 state とする。

### 8.2 emission

- learned GR similarity
- raw NCC / robust GR residual
- formation compatibility
- regional surface prior からの距離

### 8.3 transition

- `dU/dMD` の物理範囲
- slope acceleration penalty
- branch jump penalty
- known prefix との接続制約
- trajectory geometry と regional dip prior

### 8.4 inference

- Viterbi 1-best だけでなく K-best paths を保持
- forward-backward または particle/lattice posterior を計算
- mode probability、entropy、credible interval を出力
- RMSE 用 prediction は posterior mean

PF の seed luckを減らすため、可能なら deterministic dynamic programming を主実装とし、particle approximation は比較対象にする。

### 8.5 重要な ablation

- raw GR emission vs learned emission
- regional prior あり/なし
- 1-best vs posterior mean
- K 値と state grid 間隔
- transition smoothness
- uncertainty calibration

## 9. Model D: residual sequence model

path posterior が作った base に対し、残った局所・低周波誤差だけを学習する。

### 9.1 target

```text
residual = true_TVT - lattice_posterior_mean_TVT
```

必ず外側 fold の OOF base prediction から作る。in-fold base prediction を残差 target に使わない。

### 9.2 入力

- posterior mean / std / entropy
- top-2 mode distance と probability
- emission margin
- transition cost
- regional prior residual
- GR/typewell local features
- `U` の slope/curvature
- MD since prefix、suffix fraction

### 9.3 候補

- residual CatBoost/HGB
- 1D TCN
- BiGRU は TCN との比較

TCN 初期構成:

- 32--64 channels
- 4--6 residual blocks
- kernel size 5
- dilation 1--32
- dropout 0.1--0.2
- well sequence batching

補正には 2--4 ft 程度の cap と prefix からの fade-in を持たせる。

## 10. Model E: tail-risk model

RMSE は少数の catastrophic wells に強く支配される。平均的な井戸をさらに0.1改善するより、branch を外した井戸を救う方が5点台への効果が大きい。

### 10.1 risk predictor

OOF から well-level failure probability を予測する。

- posterior entropy
- top-2 branch mass
- path disagreement
- known-prefix backtest error
- spatial distance
- typewell mismatch
- eval length / Z span

### 10.2 risk-aware action

高リスク well では方向付き offset を推測しない。

- posterior を広げる
- branch mixture の posterior mean を使う
- regional prior の weight を増やす
- correction cap を小さくする

selector はOOFで固定し、LBを見て閾値を変更しない。

### 10.3 loss

公式 pooled RMSE を中心にしつつ、学習時には以下を比較する。

- row RMSE
- well-balanced RMSE
- Huber + row RMSE fine-tune
- worst-well CVaR regularization

tail objective が pooled RMSE を悪化させる場合は採用しない。

## 11. 自己教師あり事前学習

label付き TVT だけで alignment encoder を学習すると過学習しやすい。必要に応じて次を追加する。

- GR masked reconstruction
- adjacent-window order prediction
- horizontal/typewell contrastive alignment
- trajectory-aware augmentation
- gain/offset/noise augmentation
- local MD stretch/compression augmentation

GR augmentation は known prefix で観測される instrument gain/offset 分布に合わせる。地質 branch を変えてしまう強い warp は使用しない。

## 12. 実験 Stage

### F0: validation and pseudo-cut generator

- fixed well folds
- spatial/typewell folds
- multi-cut samples
- hidden-target invariance tests
- experiment ledger

完了条件: 同一 run が Colab とローカル smoke test で同じ fold/hash を出す。

### F1: independent delta-U surface model

- CatBoost/HGB
- fold-local spatial features
- uncertainty proxy

完了条件: current Stage 4 OOF から明確に改善し、spatial holdout でも方向が一致する。

### F2: learned emission model

- Siamese CNN/TCN
- hard-negative mining
- held-out prefix alignment benchmark

完了条件: raw NCC より true-offset rank と hard-negative margin が改善する。

### F3: deterministic K-best lattice

- regional prior + learned emission
- posterior mean
- uncertainty reports

完了条件:全5 foldsで既存 PF familyを改善し、seed varianceを大幅に減らす。

### F4: residual tree

- OOF lattice residual
- capped/faded CatBoost/HGB correction

完了条件: paired bootstrap CI 上端が0未満、tail悪化なし。

### F5: residual TCN

- fold-wise sequence training
- tree correctionとの error correlation 評価

完了条件: lattice+treeに追加して改善する。TCN単体の良さだけでは採用しない。

### F6: risk-aware posterior

- ambiguity calibration
- high-risk branch mixture
- well-level CVaR診断

完了条件: worst 5/10% SSE share と pooled RMSE の両方を改善する。

### F7: full independent ensemble

- surface prior
- lattice posterior
- residual tree/TCN
- physics projection
- nonnegative OOF blend

完了条件: GroupKFold、spatial holdout、typewell holdout、複数seedの全 gate を通過する。

### F8: all-train package and Kaggle inference

- fold artifacts
- all-train models
- feature builder
- model manifest
- standalone Internet-OFF Notebook
- submission audit

## 13. 到達マイルストーン

Public LB とローカル OOF の絶対値には固定対応がないため、LB 予測を数式的に換算しない。進捗は次の順で判断する。

### Milestone 1: 独自系が現行ローカル系を超える

- current Stage 4 OOF を全体・fold・tailで改善
- 公開 pretrained prediction を使わない

### Milestone 2: 独自系が公開 lineage と異なる価値を持つ

- Kaggle上で単独提出
- safe public baseとのprediction/error correlationを測定
- 小blendでなく単独でも競争力を持つ

### Milestone 3: 6点台前半

- independent model単独または独自componentだけのensembleで6.x
- rerun/seed安定性を確認

### Milestone 4: 5点台

- Public LB `< 6.000`
- LB専用offsetやsample-specific overrideなし
- validation上の改善要因を説明可能

5点台が一度出ても、同一コードのrerun分布とprivate splitを考慮し、安定成功とは別扱いにする。

## 14. 採用 gate

候補 component は以下をすべて満たす。

- hidden target invariance
- fold-local preprocessing / imputer / retrieval
- 5 folds中4 folds以上で改善
- paired well bootstrap 95% CI 上端 `< 0`
- spatial holdout で改善方向
- typewell/region holdout で catastrophic failure を増やさない
- worst 10% SSE share を悪化させない
- 3 seeds で改善方向が一致
- correctionを外したablationとの比較あり
- train/test feature schema hash一致

## 15. 優先順位を下げるもの

- 最初から大型Transformerを end-to-end 学習
- path constraintなしのcross-attention直接TVT回帰
- row random split
- validation wellを含むglobal spatial imputer
- raw DTWの1-best pathをそのまま提出
- 多数のpost-processを同時に追加
- Public LBでthreshold/offsetを選ぶ
- 5.xというDiscussion申告値からarchitectureを推測する
- 公開モデルを混ぜた結果を独自モデルの進捗として数える

## 16. 計算環境

### CPU前処理

- well CSV parsing
- multi-cut generation
- spatial index
- lattice candidate construction
- OOF report

をcacheし、model sweepごとに再計算しない。

### GPU学習

- emission CNN/TCN
- residual TCN
- self-supervised pretraining

小型sequence modelはまず単一GPUで再現性を優先する。multi-GPUはdata loaderとfold実行の並列化が安定してから使用する。

### artifact

各runで次を保存する。

```text
data/fold/schema hash
pseudo-cut configuration
model config and seed
fold and all-train checkpoints
OOF predictions
well-level diagnostics
alignment/lattice diagnostics
bootstrap result
runtime and memory
```

## 17. 直近の着手順

6点台Track Aと並行して、5点台Trackは次の順で開始する。

1. multi-cut、spatial fold、typewell foldを実装
2. targetをTVTから`U=TVT+Z`へ切り替えた独自baselineを作る
3. fold-local regional surface modelを作る
4. raw NCCのalignment benchmarkを固定する
5. learned emission Siamese TCNを実装する
6. deterministic K-best latticeを実装する
7. OOF lattice residualにtree/TCNを追加する

最初の実装対象は大型ニューラルモデルではなく、`validation/multi-cut基盤 + delta-U surface baseline` とする。ここが正しくなければ、後続の高度なsequence/path modelのscoreを信用できない。

## 18. 現実的な判断

5点台は、公開モデルを使わない独自研究としては高難度である。ただし、公開 lineage の主な弱点は明確である。

- GRの反復によるbranch ambiguity
- PF seed variance
- regional low-frequency surfaceの推定不足
- catastrophic wellsへのRMSE集中
-似たPF/GBDT同士の高い誤差相関

本戦略はこれらを、learned emission、deterministic posterior lattice、fold-local regional prior、risk-aware residual modelで直接狙う。成功可能性を上げる鍵は、architectureの大きさではなく、正しい疑似test、確率的branch処理、空間リークのない検証である。

## 19. Implementation status

Stage 11 implements F0 and the first F1 baseline in `notebooks/250_run_stage11_multicut_delta_u.ipynb`.

- four fixed pseudo-test cuts per training well;
- ordinary well, six-block spatial, and typewell-signature holdouts;
- hidden-suffix target invariance audit;
- independent `U = TVT + Z` HGB surface prior;
- fold-local leave-one-well-out regional features;
- fixed 0.35 correction shrinkage and diagnostic-only weight grid;
- OOF coefficients, row predictions, tail metrics, bootstrap, schema hash, and environment artifacts.

The model is validation-only and uses no public prediction. A full 773-well pass decides whether F2 raw-NCC/learned-emission work starts immediately or F1 requires another iteration.

The full Stage 11 run improved ordinary OOF by 4.0877, all five folds, spatial holdout by 3.0359, and typewell holdout by 3.3680. Stage 11C now performs nested weight/cap selection and replaces the relative worst-10% share veto with absolute-tail SSE/CVaR/P90 gates before F2 begins.
