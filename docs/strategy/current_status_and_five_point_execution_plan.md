# ROGII 現状・5点台到達実行計画（引き継ぎ正本）

最終更新: 2026-07-23
目標: Kaggle Leaderboard RMSE `< 6.000` を、再実行可能なInternet-OFF Notebookで達成する。  
位置付け: 以後のセッションは、まずこの文書を読み、ここから再開する。過去の戦略文書より本書の実測値と判断を優先する。

## 1. 結論

現時点で再現できた安全な最良スコアは `6.589` である。これは470 top-PF安全版で、旧best `6.685`を`0.096`改善した。Stage 18 ranked retrievalはKaggle時間制限を2回超過してスコアが付かず、独自Stage 15は `35.110` で失敗した。

5点台へ最短で到達するため、次の順序へ切り替える。

1. `6.589` の470 top-PF安全版を提出controlとして凍結する。
2. Stage 16B v003の短prefix pseudo-testと4種類のgroup holdoutを検証基盤にする。
3. Kaggle上の重いdonor探索をやめ、独自補正を学習済み低次元モデルへ圧縮する。
4. OOFでstandard/spatial/typewell/branch-groupとtailをすべて改善したモデルだけをInternet-OFF packageへ進める。
5. LB `6.3 -> 6.15 -> 5.999未満` の順で到達する。公開LB由来の特定well補正や係数だけを変えた大量提出は行わない。

Stage 15のpackage・Internet-OFF・manifest・提出監査の仕組みは再利用するが、Stage 15の予測モデルとCV値は採用しない。

## 2. 実測スコア台帳

低いほど良い。公開Notebookのタイトル・作者表示値ではなく、このアカウントで実際に得た値を判断に使う。

| 系列 | 実測LB | 判断 |
|---|---:|---|
| 改善前の公開Notebook | 7.155 | 初期control |
| ROGII public MHA safe submission build | 6.997 | 7点突破control |
| Stage 7系public blend | 7.115 | 不採用 |
| Stage 10C alignment版 | 6.994 | MHA controlと同程度 |
| public frontier MHA250SEP2 sanitized | 6.874 | 改善したが凍結base未満 |
| V599 A130 branch-conservative系 | 6.768 | 中間到達点 |
| V599 A130 branch-conservative sanitized frontier | 6.685 | 旧safe best |
| 「6.594」branch-overlap frontier sanitized | 6.693 | 表示値を再現せず、不採用 |
| top-PF A130 branch-conservative target-safe 470 | **6.589** | 現在の凍結best |
| V599 + Stage 18 ranked retrieval | スコアなし | Kaggle時間超過2回、提出経路停止 |
| Stage 15 fold-safe independent | 35.110 | 重大失敗、追加提出禁止 |

重要な差分:

```text
6.589 - 6.685 = -0.096（safe improvement）
6.589 - 5.999 =  0.590（5点台までの必要改善）
```

470でPF GR likelihood scale `×1.3`のsafe improvementは確認できた。一方、申告6.478は6.49 sourceと完全一致で実行揺らぎ、申告6.390は特定public wellへのLB-derived補正だった。5点台には独自の一般化可能な学習信号が必要である。

## 3. 主要Notebookと役割

### 提出control

- `notebooks/470_kaggle_top_pf_a130_branch_safe.ipynb`
  - 現best `6.589` に対応する凍結control。
  - PF GR likelihood scale `×1.3`と保守branchを持ち、same-well target transferを除外。
- `notebooks/230_kaggle_v599_a130_frontier_safe.ipynb`
  - 旧best `6.685` に対応する比較control。
- `notebooks/240_kaggle_branch_overlap_6594_safe.ipynb`
  - 表示上6.594だったbranch-overlap版。手元実測は`6.693`。
  - 230との差分解析対象。提出baseにはしない。
- `notebooks/220_kaggle_public_frontier_safe.ipynb`
  - MHA250SEP2系の比較対象。

### 独自研究（旧Stage 11--15）

- `notebooks/250_run_stage11_multicut_delta_u.ipynb`
- `notebooks/260_run_stage11c_delta_u_robust_gate.ipynb`
- `notebooks/270_run_stage12a_raw_ncc_benchmark.ipynb`
- `notebooks/280_run_stage12b_learned_emission_tcn.ipynb`
- `notebooks/290_run_stage12c_spatial_kbest_lattice.ipynb`
- `notebooks/300_run_stage13_emission_uncertainty_gate.ipynb`
- `notebooks/310_run_stage14_crossfit_emission_residual.ipynb`
- `notebooks/320_run_stage14b_extended_residual_gate.ipynb`
- `notebooks/330_build_stage15_fold_safe_package.ipynb`
- `notebooks/340_kaggle_stage15_internet_off_inference.ipynb`

この系列は実装資産として保持するが、Stage 14B CV `11.8047`がLBを予測できず、Stage 15が`35.110`だったため、現在の昇格判断には使用しない。

### 過去の詳細文書

- `docs/strategy/five_point_independent_strategy.md`: 独自モデルの長期設計
- `docs/strategy/six_point_execution_strategy.md`: 6点台到達時のpublic/residual計画
- `docs/strategy/public_top_model_dependency_analysis.md`: 公開model package分析
- `docs/strategy/stage15_fold_safe_inference.md`: Stage 15 package手順（モデル性能は失敗）

## 4. これまでに分かったこと

### 4.1 有効だったこと

- public pretrained/PF/MHA/V599 lineageは、実testで6点台を安定して出す。
- branch-conservative処理は、少なくとも手元では6.685まで到達した。
- Internet-OFF推論、外部Dataset package、submission順序・finite監査は構築済み。
- Stage 12Bのlearned emissionは旧疑似cut上でraw NCCのrankを大幅に改善した。
- Stage 14B residualは旧CV上ではstandard/spatial/typewellの全系列を改善した。

### 4.2 無効または信用できなかったこと

- 公開Notebookタイトルのスコアを、そのまま再現値として扱うこと。
- Stage 11--14のmulti-cut CVからLB絶対値を換算すること。
- `0.35/0.50/0.65/0.80`中心のcutで、実testの短いprefixを代表させること。
- well完全holdoutだけを最重要視すること。実testはbranch/trajectory overlap構造を持つ可能性が高く、問題設定を難しくし過ぎた。
- 最大256点へ間引いたsuffix評価だけで、提出全行の品質を保証すること。
- OOFで良いsurface/emission/residualを、そのままtestへ移せると仮定すること。
- 小さなpostprocess、weight、cap変更をLB改善の主経路にすること。

### 4.3 Stage 15失敗の意味

Stage 15は次を厳格化した。

- testと同じIDのtrain wellをheld-outにする。
- 未知IDではfold ensembleを使う。
- hidden TVTを推論featureから除外する。

それでもLBは`35.110`だった。これは単一バグと断定できないが、少なくとも旧CVと実testの生成過程・prefix長・overlap条件が一致していなかったことを示す。Stage 15をweight調整して再提出してはならない。

## 5. 改訂した中心仮説

5点台への主要改善源は、平均的なrowを少し改善することではなく、誤った地層branchを選んだ坑井・区間の大誤差を減らすことである。

```text
strong public base
  + target-free branch-overlap/retrieval signal
  + test-like prefix validation
  + branch uncertaintyとconservative fallback
  + OOF residual（有効な場合のみ）
```

`branch-overlap`は「同一test IDから正解TVTを直接引く」ことを意味しない。採用可能なのは、軌跡、X/Y/Z、GR、typewell、既知prefixなど推論時に利用可能な入力から学習・検索でき、疑似testのvalidation targetを特徴生成時に見ない処理だけである。

## 6. 新しい検証原則

### 6.1 実test prefix比率を再現する

公開test 3坑井の既知prefix比率は概ね次である。

```text
000d7d20: 1442 / 5278 = 0.273
00bbac68: 1545 / 7559 = 0.204
00e12e8b: 2083 / 6384 = 0.326
```

新しい疑似testは少なくとも`0.18--0.35`を重点化する。推奨cut grid:

```text
0.18, 0.22, 0.26, 0.30, 0.34
```

補助診断として`0.40, 0.50, 0.65`も残すが、主昇格指標には短いprefixを使う。

### 6.2 suffix全行を採点する

- 間引きは学習時のmemory節約にのみ使用可能。
- 昇格RMSEはcut以降の全行で計算する。
- sample weightは公式と同じrow pooled RMSEを主とする。
- well RMSE、p90、max、worst 10% absolute SSEも保存する。

### 6.3 三種類のholdoutを分離する

1. `branch-overlap holdout`: 近接branchはtrain側に残し、validation targetだけ隠す。実testに近い主指標。
2. `group holdout`: validation well全体を除外。未知坑井fallback診断。
3. `spatial holdout`: 周辺領域を除外。catastrophic risk診断。

主選択は1、リスク監査は2/3とする。旧Stage 15のように2/3だけで提出仕様を決めない。

### 6.4 target leakage防止

- validation suffix TVTをfeature builderへ渡さない。
- validation IDをキーに完成TVTをlookupしない。
- donor targetを使う学習・retrievalはouter foldのtrain側だけで構築する。
- 既知prefix TVTは利用可能。
- 各rowに`source_fold`, `donor_ids`, `target_visible`を保存できる設計を優先する。
- hidden-target invariance testを全feature branchに実装する。

### 6.5 controlを同じ疑似testで評価する

最低限、次を同じcut・同じ全suffix行で比較する。

- last-known/anchor extrapolation
- V599 A130相当base
- branch-overlap 6.693版
- delta-U surface
- learned emission
- 追加する新component

新validationがpublic pipelineを不自然に最下位へ置く場合、そのvalidationを提出判断に使わない。

## 7. 実行ロードマップ

## Stage 16A: 6.685と6.693の完全差分監査

目的: 表示上強いNotebookを追うのではなく、実測bestとの差分を機能単位で把握する。

対象:

- `230_kaggle_v599_a130_frontier_safe.ipynb`
- `240_kaggle_branch_overlap_6594_safe.ipynb`
- 各Notebookが参照するKaggle Dataset/model version

実装内容:

- code cellを正規化してdiffを出す。
- model/input Dataset一覧、version、path解決順を抽出する。
- raw branch予測、blend前、projection後、postprocess後の各段階を保存する。
- 3公開test坑井ごとに予測差のmean/std/max、符号、発火row数を出す。
- seed・GPU・非決定演算を記録する。
- target transfer、probe、LB bias、sample-specific overrideがないことを再監査する。

成果物:

```text
artifacts/stage16a_frontier_diff/
  notebook_diff.md
  dependency_manifest.json
  stage_prediction_differences.parquet
  per_well_difference.csv
  sanitation_audit.json
```

完了条件:

- 6.685版と6.693版の差が、名前ではなく計算グラフとして説明できる。
- 6.685版の再現に必要な入力Dataset versionを固定できる。
- 次Stageでtrain replay可能なbranchと、test専用branchを分類できる。

Kaggle提出: なし。

## Stage 16B: test-like branch-overlap疑似test基盤

目的: 旧CVを置き換え、実testと相関する評価器を作る。

実装内容:

- train坑井から`0.18--0.35`中心のprefix/suffixを生成する。
- suffix TVT列を物理的に別frameへ分離し、feature builderへ渡せない構造にする。
- X/Y/Z軌跡の近接区間、GR相関、typewell similarityからbranch overlap候補を構築する。
- branch family/groupを作り、branch-overlap holdoutを定義する。
- 全suffix rowのcontrol RMSEとtail reportを保存する。
- 同じwellから複数cutを作ってもouter splitは共有する。

成果物:

```text
artifacts/stage16b_testlike_validation/
  pseudo_test_manifest.parquet
  branch_groups.parquet
  donor_graph.parquet
  folds.parquet
  control_predictions.parquet
  control_metrics.json
  hidden_target_invariance.json
```

必須gate:

- hidden-target invarianceが全件合格。
- suffix全行で採点。
- 同一設定を再実行してID/hash/foldが一致。
- 短prefix帯ごとのRMSEが出る。
- branch-overlap、group、spatialの三指標を混同しない。

Kaggle提出: なし。

## Stage 17: V599/public strong-baseのtrain replay

目的: 6.685の強いbaseをtrain疑似test上へ再現し、改善targetを作る。

実装内容:

- public packageに正規OOFがあるbranchはprovenanceを確認して使用する。
- OOFがないpretrained branchは、outer fold内で再学習できるものだけ評価対象にする。
- test-only codeは無理にCVへ偽装せず、`unvalidated`と明記する。
- 各疑似cutでbase prediction、候補path、uncertainty、中間branchを保存する。

target:

```text
residual_target = true_tvt - strong_base_tvt
```

完了条件:

- 全suffix行のstrong-base OOFが得られる。
- branch-overlap主検証でanchor/delta-U独自baseより明確に良い。
- fold provenanceを機械的に監査できる。

Kaggle提出: 原則なし。どうしても推論再現確認が必要な場合のみcontrol 1本。

## Stage 18: target-free branch retrieval / overlap corrector

目的: 5点台へ必要な新しい信号を追加する。最優先model stage。

候補feature:

- X/Y/Z/MD trajectory segment distance
- azimuth、inclination、curvature、horizontal displacement
- known-prefix `U = TVT + Z` level/slope
- horizontal GR multi-scale normalized correlation
- typewell GR/formation signature
- donor branchとの重複長・距離・方向
- top-1/top-2 donor margin
- donor間prediction disagreement

候補model:

1. deterministic nearest-segment retrieval + robust weighted average
2. LightGBM/CatBoostのdonor ranking
3. pairwise Siamese trajectory+GR encoder
4. retrieval predictionをpriorにしたsmall residual HGB

予測対象は絶対TVTより`delta-U`またはstrong-base residualを優先する。

必須ablation:

- geometry only
- GR only
- geometry + GR
- typewell追加
- donor target prior追加
- uncertainty gateあり/なし

昇格条件:

- branch-overlap主検証でpooled RMSE `-0.30`以上を第一目安とする。
- 5分割中4分割以上改善。
- bootstrap 95% CI上端`< 0`。
- worst-tail absolute SSEとwell maxを悪化させない。
- group/spatial holdoutでcatastrophic悪化がない。

Kaggle提出候補1: 6.685 base + Stage 18 correction。目標LB `<= 6.40`。

## Stage 19: alignment/emissionのtest-like再学習

目的: Stage 12Bの有効なrank学習を、短prefix・branch overlap条件へ合わせ直す。

変更点:

- cut分布を`0.18--0.35`中心へ変更。
- 学習negativeに近接branch/repeated-GR hard negativeを入れる。
- inferenceは全解像度、または補間誤差を明示測定する。
- surface誤差を含む候補gridで学習する。
- raw NCC、learned logits、retrieval priorを別channelとして保持する。
- posterior meanとbranch mixtureを比較する。

昇格条件:

- Stage 18 baseへ追加して`-0.15`以上。
- top-k recallだけでなく全suffix RMSEが改善。
- high-entropy区間でconservative fallbackが有効。
- catastrophic wellsを増やさない。

Kaggle提出候補2: Stage 18 + Stage 19。目標LB `<= 6.15`。

## Stage 20: residual・tail rescue・最終5点台package

目的: 残る系統誤差と少数のbranch failureを減らし、`< 6.000`へ到達する。

residual feature:

- strong-base/retrieval/emissionのprediction disagreement
- posterior entropy、top-2 separation、donor margin
- prefix backtest error
- MD since cutoff、suffix fraction
- `U` slope/curvature/roughness
- branch overlap lengthと距離

model:

- cross-fitted CatBoost/HGBを先に試す。
- decorrelated gainがある場合のみsmall TCNを追加する。
- correction weight/capはnested OOFで固定する。
- high-riskでは大補正を避け、branch mixtureまたはbaseへ戻す。

最終昇格条件:

- branch-overlap検証でStage 19より改善。
- group/spatial監査で大崩れしない。
- tail absolute SSE、p90、maxが非悪化。
- 係数違いではなく、予測ロジックとして説明可能。
- Internet-OFF hidden rerunに対応したpackageを作れる。

Kaggle提出候補3: full stack。目標LB `< 6.000`。

## 8. 最短化のための優先順位

### 今すぐ行う

1. Stage 16A差分監査
2. Stage 16B test-like validation
3. Stage 17 strong-base replay
4. Stage 18 branch retrieval

### 後回し

- 大型Transformer
- 多GPU分散学習
- Stage 15のweight/cap再探索
- 6.685/6.693の微小係数探索
- public Notebookの表示スコアだけを根拠にした提出
- residualを先に大規模化すること

最短経路の核心は、Stage 18のbranch retrievalを早く正しい検証へ載せることである。

## 9. 提出ルール

- 凍結controlは6.685。
- LB差`< 0.02`を根拠に次の係数を選ばない。
- 同一ロジックのweight違いを連続提出しない。
- 新しい提出は原則としてローカル主検証`-0.15`以上、または明確なcatastrophic rescueを必要とする。
- Notebook title、Git commit、Kaggle Dataset slug/version、submission hash、LB scoreを台帳へ追記する。
- 失敗提出も削除せず、原因と再提出禁止条件を記録する。

推奨台帳列:

```text
date
git_commit
notebook_name
kaggle_notebook_version
input_dataset_versions
submission_sha256
local_primary_rmse
local_delta_vs_control
public_lb
status
decision
```

## 10. 計算環境と配置

### ローカル/GitHub

```text
repository: https://github.com/Okada-N13/rogii.git
branch: main
workspace: C:\Users\owner\kaggle\ROGII
```

本書作成直前の実装HEADは`5243560`。本書を追加したcommit以降は`git log -5 --oneline`で確認する。

### Colab/Drive

```text
/content/ROGII
/content/drive/MyDrive/kaggle/rogii/data
/content/drive/MyDrive/kaggle/rogii/artifacts
```

- tree/前処理/差分監査: CPUでよい。
- TCN/emission: T4 GPU。
- 現在のKaggle PyTorchではP100が`no kernel image`になったため使わない。
- T4 x2でも現行コードが使うのは基本1枚。

### Kaggle

- Internet: OFF
- Accelerator: T4 x2
- Dataset名をコードへ固定せず、manifestを再帰検出する。
- v001/v002など複数packageを同時にAdd Inputしない。
- hidden testのログは取得できない前提。成功/失敗とLBだけで判断する。

## 11. 再利用する実装資産

- `src/rogii/data/multicut.py`: prefix/suffix生成、target-free inference record
- `src/rogii/models/delta_u_surface.py`: delta-U surface
- `src/rogii/models/emission_features.py`: NCC/emission入力
- `src/rogii/models/emission_tcn.py`: candidate-shared TCN
- `src/rogii/models/emission_residual.py`: residual feature/model
- `src/rogii/cli/stage15_package.py`: manifest/hash/package構築の雛形
- `src/rogii/cli/stage15_infer.py`: Internet-OFF推論・submission auditの雛形

再利用するのはコード基盤であり、Stage 15の学習済み重みや`generic_w080_cap16`を次の提出へ自動採用しない。

## 12. 5点台のDefinition of Done

最低条件:

- Kaggle表示スコアが`< 6.000`。
- hidden rerunが正常終了。
- Internet-OFF Notebookから`submission.csv`を生成。
- probe、LB-derived offset、sample-specific target overrideなし。
- Git commit、Notebook version、Dataset version、submission SHA-256を保存。

望ましい安定条件:

- 同じNotebookのrerunで大崩れしない。
- 疑似testの改善要因とLB改善方向が一致する。
- 5点台が単一坑井overrideだけに依存しない。
- private/generalizationに備え、group/spatial holdoutでcatastrophic failureが管理されている。

## 13. 新しいセッションでの再開手順

新しい担当者・Codexセッションは次を順に行う。

1. 本書を最後まで読む。
2. `git status --short`でユーザー変更を確認し、勝手に破棄しない。
3. `git log -10 --oneline`で本書以降の変更を確認する。
4. 実測bestが`6.589`から更新されていないかユーザーへ確認する。
5. `notebooks/230...`と`240...`、参照Dataset情報を確認する。
6. Stage 15を再提出しない。`35.110`は失敗として固定する。
7. 未完ならStage 16Aから開始する。
8. 各Stage実装後、Colab用Notebookを単独実行可能にする。
9. ユーザーから結果を受け取るまで次Stageを勝手に昇格しない。
10. 実測値・判断・commitを本書へ追記する。

再開時にユーザーから最低限必要な情報:

- 新しいLBスコアがあるか
- どのNotebook/Dataset versionを実行したか
- Colab artifactがDriveに残っているか

非公開testログは要求しない。Kaggle側から取得できない前提で設計する。

## 14. 現在の次アクション

Stage 16Aは完了した。230/240の上流codeとdependencyは同一で、唯一の予測差は最終midpoint hedgeだった。詳細は`docs/strategy/stage16a_frontier_diff_report.md`を参照する。

Stage 16Bもローカルfullで完了した。773 wells、6,184 cuts、26,225,067 suffix rowsを固定した。環境非依存v003のmanifest hashは`af748e7092b8a605a756f478ebad80a95286631d8214a32d48f6af59b82b579c`で、fold、donor構造、branch groupにも独立hashを持たせた。詳細は`docs/strategy/stage16b_testlike_validation_report.md`を参照する。

Stage 16B v003はローカル/Colabで4 hashが完全一致し、固定manifestとして確定した。

Stage 17Aは全gateを通過した。primaryの50.063%で公開OOFをtarget-safeに再利用でき、eligible RMSEは`14.738 → 11.354`、未coveredをlast-knownとしたfull hybridも`23.096 → 22.118`へ改善した。両指標とも5/5 fold改善である。

Stage 17Bも全gateを通過した。uncovered primary RMSEは`29.162 → 17.123`、Stage 17Aを含むfull primaryは`22.118 → 14.524`で、5/5 fold改善した。ただしfold 3の改善は`-0.119`と小さい。

Stage 17C gateは棄却した。always-selector `14.524`に対してgateは`15.069`（`+0.545`）、4/5 foldとP90を悪化させ、bootstrap 95%も`[+0.0668,+0.5367]`だった。threshold gridも全滅のためStage 17B always-selectorを維持する。

Stage 17Dは全gateを通過した。medium/highともbaseline比で大幅改善し5/5 fold、screen比も全体でそれぞれ`-0.834`、`-2.577`だった。Stage 17B always-selectorをvalidation controlとして凍結し、Stage 17を完了する。

Stage 18A全体gateは棄却した。standard primaryは`18.565 → 16.356`（`-2.209`）だが4/5 folds、cut P90悪化、bootstrap上限が正だった。spatial除外は近接donorが32/150 cutsにしか残らず`+3.707`悪化した。一方、branch-group除外は`18.565 → 15.850`（`-2.715`）、5/5 folds、cut P90非悪化だった。

Stage 18Bは全gateを通過した。Stage 18Aと重複0の独立150 cutsでbranch-group primaryは`12.010 → 10.399`（`-1.611`）、5/5 folds、cut P90 `-2.223`、bootstrap 95% `[-1.317, -0.339]`だった。

Stage 18Cも全gateを通過した。全3,865 primary cuts・18.84M rowsで`14.524 → 12.784`（`-1.740`）、coverage 100%、5/5 folds、5/5 prefix fractions、P90 `-1.229`、bootstrap 95% `[-1.125, -0.680]`だった。固定top-4 branch retrievalを新controlとして凍結する。

Stage 18Dも全gateを通過した。固定retrieval比`12.784 → 12.645`（`-0.139`）、5/5 standard folds、4/5 branch folds、5/5 fractions、P90・max改善、bootstrap 95% `[-0.205, -0.0099]`だった。rankerを昇格する。

Stage 18E packageは完成した。5 fold-safe rankers、43,758 candidate rows、same-well target leakage guardを含み、manifest SHA-256は`7bddc1914f3d046b678dbb8f5d1cc17427b03bc85c1a06d1f2088cbe68d3935d`である。

Stage 18Fは終了した。Kaggle用Notebook`notebooks/460_kaggle_v599_stage18_ranked_retrieval.ipynb`はhidden rerunで2回時間超過し、LBスコアを取得できなかった。予測品質とは別に提出可能性を満たさないため、この経路を停止する。

Stage 18F v001はinteractive auditを通過したがsubmission rerunが時間超過し、LBスコアは付かなかった。v002では予測を変えず、1,546個のdonor CSV再読込を単一NPZ cacheへ置換した。

v002 public placeholder実行でStage 18は26.30秒だった。Kaggle hidden rerunは約200 wellsへ拡大するため、3-well固定監査を除去し、donor KD-treeをwell間で再利用するv003へ更新した。v003 public placeholder監査は3/3 wells applied、fallback 0、26.51秒で通過したが、hidden rerunは時間超過した。

並行して、申告スコア6.49の公開`top-pf-config-branch-conservative(1).ipynb`を監査した。6.685 controlに対するhidden-test-activeな主要差分はlearned PFのGR likelihood scale `×1.3`で、branch hedgeはstrength `0.60`、cap `2.0`、既存routeをskipしない設定だった。公開3 fake wellsだけに効くsame-well train TVT/contact transferは採用しない。安全再現版を`notebooks/470_kaggle_top_pf_a130_branch_safe.ipynb`として用意し、Stage 18を混ぜない独立候補とする。詳細は`docs/top_pf_649_safe.md`を参照する。

Stage 18F v003は再度Kaggle時間制限を超過し、スコアが付かなかった。Stage 18を現在のV599へ追加する構成は停止する。470安全版は`6.589`で、旧safe best `6.685`を`0.096`改善した。申告6.49/6.478との差は安全に再現できていない。

Stage 19Aは全gateを通過した。固定weight `0.50`・cap `16 ft`でstandard `-0.7278`、spatial `-0.5587`、typewell `-0.4627`、branch-group `-0.7443`。standard/typewell/branchは5/5 folds、spatialは5/6 folds、全5 prefix fractions、P90、worst-tail、bootstrapを改善した。

Stage 19Bは全gateを通過した。3,865 cuts、773 wells、66特徴、15 HGB modelsで、特徴parity最大差`0.0`。773 wellsは611.915秒、hidden 200 wells推定158.32秒だった。manifest SHA-256は`5bf1c84b...e47d632`。

Stage 19CはKaggle interactive監査を正常通過したが、Public LBは`6.958`で6.589 controlから
`+0.369`悪化した。Stage 19Cを棄却し、weightだけを下げた再提出は行わない。

Stage 20Aはprimary gate不通過だった。固定weight 0.10は`9.9810 → 9.8611`で5/5 folds、
5/5 fractionsを改善したが、bootstrap上限`+0.00625`、well P90`+0.1006`で不合格。
weight 0.50は2/5 foldsしか改善せず、Stage 19CのLB悪化と整合した。

Stage 20Bもprimary gate不通過だった。Stage 20Aの158 wellsと重複ゼロの139 wellsで、
weight 0.05はstandard `-0.02956`、5/5 folds/fractions、P90改善だったが、
bootstrap上限`+0.00350`、spatial `+0.00921`、typewell `+0.02826`で不合格。
3係数trajectory residualを終了し、Stage 20Cは実施しない。

Stage 21Aは77 cuts・63 wellsで完了したが棄却した。base `9.1901`に対してguarded routerは
`9.8522`（`+0.6621`）、bootstrap 95% `[+0.1895,+1.7333]`、standard改善1/5だった。
internal/outer rank correlation `0.2754`に対しtop-1 oracle一致率は`5.19%`しかなく、
2 internal cutsの最小scoreで候補を直接選ぶ規則は一般化しなかった。raw router
`450.64`を生んだ多項式候補は今後使用しない。一方oracle `5.4550`の候補多様性は確認できた。

Stage 21Bも棄却した。Stage 21Aと重複ゼロの62 cuts・58 wellsで、base `8.90381`に対し
primaryは`+0.00945`、weight 0.05も`+0.00205`だった。bootstrap上限は正で、
standard 3/5、spatial 3/6、typewell 1/5、branch 3/5。候補別楽観バイアス補正でも
改善方向にならないため、Stage 21Cは実施せずvisible-prefix candidate routingを終了する。

Stage 22Aも棄却した。Stage 21Aの63 wellsで学習し、完全非重複のStage 21B 58 wellsで
評価したが、base `8.90381`に対しprimary weight 0.25は`+0.18158`、weight 0.10も
`+0.05757`だった。standard 1/5、spatial 1/6、typewell 2/5、branch 0/5で、
well P90も`+0.7580`悪化した。Stage 22Bやweight縮小は実施しない。

Stage 23Aは全gateを通過した。62 cuts・58 wellsでoffset coverage `99.31%`、oracle
`8.6132 → 2.2731`（`-6.3401`）、raw top10 `31.14%`対random `16.39%`、median rank 20。
standard 5/5、spatial 6/6、typewell 5/5、branch 5/5、fraction 4/4でrank信号が安定した。
raw smooth decoder自体は最良でも`+0.0234`のため採用せず、学習emissionへ進む。

Stage 23Bはrankerとして成功した。外部validationでtop10 `30.70%→70.33%`、top5
`18.60%→49.98%`、median rank `20→6`、NLL `-1.2057`、全groupで改善した。
ただし固定posterior mean decoderはRMSE `-0.0889`、bootstrap上限`+0.0345`、
P90 `+0.2636`でprimary gate不通過。TCNを棄却せずdecoderだけを修正する。

Stage 23Cは棄却した。training OOF最良の`summary_a1000`はRMSE `-0.1408`、P90
`-1.2421`だったが、worst fold `+0.2387`、bootstrap上限`+0.1968`でeligible profileは
0件だった。Stage 21Bへ補正を適用しなかったためvalidation deltaは`0.0`。TCN rankerは
維持するが、連続offsetを直接回帰するdecoder familyは終了する。

Stage 23Dも棄却した。最良profileはtraining OOFでRMSE `-0.1071`、P90 `-0.6347`、
4/5 folds改善したが、worst fold `+0.0817`、bootstrap上限`+0.0461`だった。
強正則化profileはworst fold `+0.0498`だがbootstrap上限`+0.0482`。全profile不合格のため
design validationへ適用していない。decoder hyperparameter調整は終了する。

現在のactive taskは **Stage 24A: scaled soft-ordinal emission** である。

1. A100/L4 GPU（T4可）で`notebooks/610_run_stage24a_scaled_ordinal_emission.ipynb`を実行する。
2. Stage 21B 58 wellsをdesign validationとして学習から完全除外する。
3. standard foldごと100 wells、合計500 wells・500 cutsを学習に固定する。
4. 別の120 wellsをStage 24B confirmation用に予約し、Stage 24Aでは一切使わない。
5. soft ordinal target、expected-offset Huber、offset-RMSE early stoppingで5-fold学習する。
6. primary correction weight `0.75`の事前gateだけを判定する。
7. 通過しても提出せず、予約120 wellsでStage 24Bを実施する。

Stage 24AからKaggle submissionを作らない。

Stage 24A初回実行は学習開始前にprediction length mismatchで停止した。原因はmanifest v001が
`replay_eligible=False` cutを許していたこと。v002はeligible primary cutだけを選び、
public OOF連続suffix・開始位置・長さをloaderでも二重監査する。失敗runからモデル結果は
生成されておらず、実験選択への影響はない。

v002 manifestは詳細stderrがNotebookに表示されないまま停止した。v003では一律fold quotaを
廃止し、24 confirmation wells/foldを固定したままtraining総数500をfold余剰へwater-fill
再配分する。以後manifest失敗時はstdout/stderrをNotebookへ必ず表示する。

Stage 24Aは完了したが棄却した。500 training wellsでrankはtop10 `30.70%→65.23%`、
top5 `18.60%→46.13%`、全group改善。一方、固定weight 0.75はRMSE `+0.0220`、
P90 `+0.5138`、bootstrap上限`+0.1749`。weight 0.25も`-0.0111`だけなので採用しない。
予約120 wellsは未使用。

Stage 25Aも棄却した。smooth mean最良は`-0.0110`、bootstrap上限`+0.0408`、
standard 3/5・branch 2/5。medianは`-0.0057`、Viterbiは全profile悪化。rowwise GR offset
stateを終了し、500-cut OOF再生成・予約120 wells確認は実施しない。

現在のactive taskは **Stage 26A: affine trajectory path-state audit** である。

1. CPUで`notebooks/630_run_stage26a_affine_path_state.ipynb`を実行する。
2. 開始offset×終了offsetの11×11=121 affine pathsを作る。
3. 62 design-validation cutsのraw GR cost volume上でcut-level scoreを計算する。
4. oracle gain、top-k rank、coverage、全group signalだけをgateにする。
5. raw decoder値は診断のみでprofile採用に使わない。
6. 全signal gate通過時だけ500-cut learned path rankerへ進む。
7. 学習・Kaggle submission・予約120 wellsの使用は行わない。

Stage 26A初版は計算開始前にproject venvの`torch`不足で停止した。修正版Notebookは
Colab標準Python＋repository `PYTHONPATH`で起動する。実験結果はまだ生成されていない。

## 15. 決定ログ

- 2026-07-22: V599 A130 branch-conservative sanitized frontierの実測`6.685`をbest controlに決定。
- 2026-07-22: 表示6.594 branch-overlap版の実測は`6.693`。6.685を更新しないため不採用。
- 2026-07-22: Stage 15 fold-safe independentは`35.110`。旧CVとLBの対応仮説を棄却し、追加提出を停止。
- 2026-07-22: 5点台を必須目標とし、public strong-base + test-like branch-overlap validation +独自correctorへ計画を再編。
- 2026-07-22: 次のactive taskをStage 16A frontier差分監査に設定。
- 2026-07-22: Stage 16A完了。230/240は上流同一で、6.693版に新しいbranch-overlap modelがないことを確認。active taskをStage 16Bへ更新。
- 2026-07-22: Stage 16Bローカルfull完了。短prefix primaryでlast-known TVT RMSE 23.096、constant-U 103.911、linear-U 59.808。旧delta-U検証の分布不一致を確認し、active taskをStage 17へ更新。
- 2026-07-22: Stage 16B初版hashのpandas/scikit-learn環境依存を検出。決定論的foldとcanonical byte hashへ置換し、Colab再照合用v002を作成。指標とcut構成は不変。
- 2026-07-22: v002でfold/donorのtie break差を検出。同サイズgroup、空間fold番号、同距離donorをwell IDで固定したv003へ更新。
- 2026-07-22: Stage 16B v003の4 hashがColabと完全一致。manifestを凍結し、Stage 17A public OOF replay実装へ移行。
- 2026-07-22: Stage 17A通過。primary row coverage 50.063%、eligible `-3.384 RMSE`、full hybrid `-0.978 RMSE`、双方5/5 fold改善。active taskをStage 17Bへ更新。
- 2026-07-22: Stage 17B通過。uncovered selector `-12.038 RMSE`、full primary `-7.594 RMSE`、5/5 fold改善。ただしfold 3がほぼ中立のためactive taskをStage 17C gateへ更新。
- 2026-07-22: Stage 17C gate棄却。RMSE `+0.545`、4/5 fold悪化、bootstrapも有意に悪化。always-selectorを凍結しStage 17D resolution auditへ移行。
- 2026-07-22: Stage 17D通過。medium/highはbaseline比`-19.971/-11.356`、screen比`-0.834/-2.577`、双方5/5 fold。Stage 17完了、active taskをStage 18A retrievalへ更新。
- 2026-07-22: Stage 18A全体gate棄却。standardは`-2.209`だが4/5 folds・P90・bootstrapで不合格。spatialはeligible 32/150で別問題化。一方branch-groupは`-2.715`、5/5 folds、P90非悪化のため、非重複150 cutsのStage 18B確認へ移行。
- 2026-07-22: Stage 18B全gate通過。非重複150 cutsで`-1.611`、5/5 folds、P90 `-2.223`、bootstrap上限`-0.339`。固定20% branch retrievalをStage 18C全primary cutsへ昇格。
- 2026-07-22: Stage 18C全gate通過。全3,865 cutsで`-1.740`、5/5 folds、5/5 fractions、coverage 100%、P90・max・bootstrapすべて改善。固定top-4 retrievalを凍結しStage 18D learned donor rankingへ移行。
- 2026-07-22: Stage 18D全gate通過。固定retrieval比`-0.139`、standard 5/5、branch 4/5、fraction 5/5、bootstrap上限`-0.0099`。5個のfold-safe modelを使うStage 18E inference packageへ昇格。
- 2026-07-22: Stage 18E package完成。5 fold-safe models、43,758 rows、manifest SHA-256 `7bddc191...d3935d`。active taskをStage 18F Kaggle inferenceへ更新。
- 2026-07-23: Stage 18F v001は全audit正常だがKaggle submission rerunが時間超過しスコアなし。予測式を固定したままdonor CSV再読込をpacked NPZへ置換するv002へ更新。
- 2026-07-23: v002 Stage 18は3 fake wellsを26.30秒で完了。hiddenは約200 wellsであることと、3-well固定監査がhidden失敗を起こすことを確認。任意well数監査とdonor KD-tree cacheを含むv003へ更新。
- 2026-07-23: Stage 18F v003 public監査は3/3 applied、fallback 0、26.51秒で通過。hidden rerunを開始。
- 2026-07-23: 申告6.49 top-PF公開Notebookを監査。same-well TVT/contact経路を除外し、PF GR scale `×1.3`と検証済み保守branch設定だけを230へ移植した470安全再現版を作成。Stage 18結果確定後に単独提出する。
- 2026-07-23: 470安全版の実測は`6.589`。旧best `6.685`を更新し、PF GR scale `×1.3`をsafe improvementとして確定。
- 2026-07-23: Stage 18F v003は2回目もKaggle時間超過でスコアなし。V599へrowwise donor retrievalを追加する提出経路を停止。
- 2026-07-23: 申告6.478 Notebookは6.49版とsource完全一致で、独立手法ではなく実行揺らぎと判定。申告6.390 Notebookは特定public well `00e12e8b`へLB-derived `+0.522 ft`を加えるため安全版へ不採用。
- 2026-07-23: active taskをStage 19Aへ更新。3係数trajectory residual、4 fold family、hidden-target invariance、軽量推論契約で独自学習路線を開始。
- 2026-07-23: Stage 19A全gate通過。固定profileでstandard `-0.7278`、spatial `-0.5587`、typewell `-0.4627`、branch-group `-0.7443`。active taskをStage 19B all-data bundle/runtimeへ更新。
- 2026-07-23: Stage 19B全gate通過。66特徴parity差`0.0`、hidden 200 wells推定158.32秒。active taskをStage 19C portable inferenceへ更新。
- 2026-07-23: Stage 19Cを実装。HGBをNumPy NPZへlossless変換し、実test `TVT_input`だけのstandalone特徴生成、Colab package Notebook、470統合Kaggle Notebookを追加。
- 2026-07-24: Stage 19C実測`6.958`。6.589から`+0.369`悪化のため棄却。代理base alignment仮説を棄却し、A130/SP45/projection/final blendを含むStage 20A固定200-cut screenへ移行。
- 2026-07-24: Stage 20Aは194 cutsで固定weight 0.10が`-0.1199`、5/5 folds/fractions改善したがbootstrap上限`+0.00625`・P90悪化で不合格。診断weight 0.05を別wellだけで固定確認するStage 20Bへ移行。
- 2026-07-24: Stage 20Bはdiscovery overlapゼロでstandard `-0.02956`だがbootstrap/spatial/typewell不合格。3係数残差を終了し、visible-prefix実測backtestで候補を選ぶStage 21Aへ移行。
- 2026-07-24: Stage 21Aは`+0.6621`悪化、bootstrap下限も正、standard 1/5で棄却。oracle余地はあるがtop-1一致率`5.19%`。多項式を廃止し、候補別楽観バイアスを別wellへ転送するStage 21B disjoint confidence gateへ移行。
- 2026-07-24: Stage 21Bは完全非重複wellでprimary `+0.00945`、weight 0.05も`+0.00205`、typewell 1/5。prefix routingを終了し、候補間rowwise disagreementから非線形residualを学ぶStage 22Aへ移行。
- 2026-07-24: Stage 22Aは完全非重複wellでprimary `+0.18158`、最小weightも悪化、branch 0/5、P90 `+0.7580`。rowwise residual fieldを終了し、strong-base周囲のGR offset-state信号を先に監査するStage 23Aへ移行。
- 2026-07-24: Stage 23Aはoracle `-6.3401`、top10はrandomの1.90倍、全standard/spatial/typewell/branch/fraction groupsでrank信号を確認。raw decoderは未改善のため採用せず、完全非重複validationのStage 23B learned emissionへ昇格。
- 2026-07-24: Stage 23Bはrankerとしてtop10 `+0.3962`、top5 `+0.3138`、全group改善。ただしdirect posterior decoderは`-0.0889`、bootstrap/P90不合格。TCNを固定しtraining OOFだけでdecoderをnested校正するStage 23Cへ移行。
- 2026-07-24: Stage 23Cはtraining OOF最良`-0.1408`だがworst fold `+0.2387`、bootstrap上限`+0.1968`で全profile不合格。連続decoderを棄却し、移動・方向・量を分けるCPU Stage 23Dへ移行。
- 2026-07-24: Stage 23Dは最良`-0.1071`、P90改善、4/5 foldsだがbootstrap上限`+0.0461`で不合格。77-cut decoder調整を終了し、500 training wells＋120 reserved wells、soft ordinal/expected-offset lossのStage 24Aへ移行。
- 2026-07-24: Stage 24Aは500 wellsでもrank信号を全groupで再現したが固定decoderは`+0.0220`、P90/ bootstrap不合格。予約120 wellsを温存し、固定checkpointsのtemporal pathだけを診断するStage 25Aへ移行。
- 2026-07-24: Stage 25Aはsmooth最良`-0.0110`、Viterbiは悪化し全profile不合格。rowwise offset stateを終了し、cut-level開始/終了offset trajectoryを監査するCPU Stage 26Aへ移行。
# 2026-07-24 update: Stage 26A -> Stage 27A

- Stage 26AのGR affine path stateは棄却。oracleは`-3.5085`だが、最良decoderは
  `+0.1369`、oracle rank中央値`59.5/121`、branch信号`3/5`だった。
- GR row-state / affine-path / temporal decoderの追加探索を終了する。
- 次はStage 27Aで、既知prefixのみから`U=TVT+Z`のXY局所平面を推定する。
- Stage 27Aは62 design-validation cutsのみ。予約120 wellsは未使用。
- 詳細は`docs/strategy/stage27_spatial_surface_state.md`。

# 2026-07-24 update: Stage 27A -> Stage 28A

- Stage 27Aは棄却。prefix XY planeと真の終点U変化は`0.92`以上で相関したが、
  A130 proxyへの固定補正は`+1.1638`悪化し、standard/branch/fractionは全group不合格。
- 強いbaseへ物理傾斜を無条件加算する系列を終了する。
- Stage 28Aは固定500 training wellsで、平滑化したbase残差を5-model HGB ensembleで学習する。
- 62 design-validation全gate通過時だけ、未使用の予約120 wellsをStage 28Bで一度開封する。
- 詳細は`docs/strategy/stage28_expanded_residual_field.md`。

# 2026-07-24 update: Stage 28A -> Stage 29A

- Stage 28Aは平均`-0.0696`、standard/branch各`4/5`まで改善したが、bootstrap上限
  `+0.0299`、P90悪化、spatial/typewell各`3 groups`で予約確認へ進まない。
- weight 0.30の診断値を見てprimaryを変更しない。
- Stage 29Aは同じ500 training wellsの全安全short-prefix cutsへ拡張する。
- モデル・固定weight 0.20・cap・validation gateは変更せず、fraction generalizationだけを試す。
- 不通過ならHGB rowwise residual familyを終了する。

# 2026-07-24 update: Stage 29A -> Stage 30A

- Stage 29Aは固定weight 0.20で`-0.0754`。P90/fraction/spatialは通ったが、
  bootstrap上限`+0.0000677`、typewell/branch各`3 groups`で予約確認へ進まない。
- 診断weight 0.30の`-0.1017`を根拠にprimaryを変更しない。
- HGB rowwise residual familyを終了する。
- Stage 30Aは同じ1,370 cutsと固定weightを使い、101-row平滑残差をdilated TCNで直接学習する。
- L4/A100推奨。全gate通過時だけ予約120 wellsを開封する。

# 2026-07-24 update: Stage 30A -> Stage 31A

- Stage 30Aは固定weight 0.20で`-0.0936`。standard/spatial/branch/fractionは通ったが、
  bootstrap上限`+0.0160`、P90悪化、typewell `3/5`で予約確認へ進まない。
- 診断weight 0.30は`-0.1334`だが直接採用しない。
- Stage 31Aは保存済み5 checkpointsの予測分散でweight 0.30補正をtarget-freeに縮小する。
- 再学習なし、62 cutsだけ。全gate通過時のみ予約120 wellsを開封する。

# 2026-07-24 update: Stage 31A -> Stage 32A

- Stage 31A primaryは`-0.10045`まで改善し全group consistencyを通過したが、
  bootstrap上限`+0.00959`とP90`+0.25152`で不合格。予約120 wellsは未使用。
- 事前定義済み`agreement_w030_a060`はpooled `-0.14505`で最良だった。
- Stage 32AはStage 31の4候補を保存済みcut reportだけで完全監査する。
- 候補、weight、閾値は追加せず、全gate通過候補の最良1件だけを固定する。
- Stage 32AはCPU・再学習なし・再推論なし。通過時のみStage 32Bで予約120 wellsを一度開封する。

# 2026-07-24 update: Stage 32A -> Stage 33A

- Stage 32Aは4候補すべて不合格。最良sign-agreementは`-0.1450`、standard `4/5`、
  fraction `4/4`だが、bootstrap上限`+0.00531`、cut P90 `+0.2377`。
- 予約120 wellsは未使用。rowwise TCN uncertainty profile調整を終了する。
- Stage 33Aは1,370 training cutsにfold-safe TCN予測を作り、cutごとのSSE最適補正係数を
  target-free特徴から学ぶ。
- 固定weight `0.30`、cap `8 ft`、ramp `96 rows`は変更しない。
- 62 design-validationの全gate通過時だけStage 33Bで予約120 wellsを一度開封する。
