# ROGII 現状・5点台到達実行計画（引き継ぎ正本）

最終更新: 2026-07-22  
目標: Kaggle Leaderboard RMSE `< 6.000` を、再実行可能なInternet-OFF Notebookで達成する。  
位置付け: 以後のセッションは、まずこの文書を読み、ここから再開する。過去の戦略文書より本書の実測値と判断を優先する。

## 1. 結論

現時点で再現できた最良スコアは `6.685` である。Notebook上で「6.594」と表示されたbranch-overlap版の手元実測は `6.693` であり、最良を更新しなかった。独自Stage 15は `35.110` で失敗した。

5点台へ最短で到達するため、次の順序へ切り替える。

1. `6.685` のV599 A130 branch-conservative版を提出controlとして凍結する。
2. `6.685` と `6.693` のコード・model artifact・中間予測を分解し、実際に効いているbranch-overlap処理を特定する。
3. trainから「実testと同じ短いprefix・branch overlap」を再現する疑似testを作り、Leaderboardと相関しない旧CVを置き換える。
4. 凍結したpublic baseに、OOFで検証できたbranch retrieval、alignment、residualだけを追加する。
5. LB `6.4 -> 6.15 -> 5.999未満` の順で到達する。係数だけを変えた大量提出は行わない。

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
| V599 A130 branch-conservative sanitized frontier | **6.685** | 現在の凍結best |
| 「6.594」branch-overlap frontier sanitized | 6.693 | 表示値を再現せず、不採用 |
| Stage 15 fold-safe independent | 35.110 | 重大失敗、追加提出禁止 |

重要な差分:

```text
6.693 - 6.685 = +0.008（悪化）
6.685 - 5.999 =  0.686（5点台までの必要改善）
```

`0.008`は非決定性や小さな条件差でも動き得るため、6.685/6.693間の係数探索には提出枠を使わない。5点台には別の信号またはcatastrophic branch errorの削減が必要である。

## 3. 主要Notebookと役割

### 提出control

- `notebooks/230_kaggle_v599_a130_frontier_safe.ipynb`
  - 現best `6.685` に対応するcontrol候補。
  - 次の作業開始時に、実際に提出したNotebook versionと一致するかセル・入力Dataset・出力hashを再確認する。
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
4. 実測bestが`6.685`から更新されていないかユーザーへ確認する。
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

Stage 16Bもローカルfullで完了した。773 wells、6,184 cuts、26,225,067 suffix rowsを固定し、manifest hashは`0d85e3e2842eb635a2a9231ee5086923166bcd0fa6d49d6ad87f871099968312`となった。詳細は`docs/strategy/stage16b_testlike_validation_report.md`を参照する。

現在のactive taskは **Stage 17: V599/public strong-baseのtrain replay** である。

実装開始時の具体的順序:

1. 230 NotebookのV599/PF/learned branchをtrain pseudo-cutへ適用可能なcomponentへ分離する。
2. public artifactのOOF provenanceを再監査する。
3. Stage 16B manifestを変更せずstrong-base predictionを生成する。
4. primary全suffix RMSEをlast-known TVT `23.096`と比較する。
5. replay不能なtest-only/precomputed branchを明示し、CV値へ混入させない。

Stage 17のstrong-base OOFが完成するまで新しい補正をKaggleへ投入しない。

## 15. 決定ログ

- 2026-07-22: V599 A130 branch-conservative sanitized frontierの実測`6.685`をbest controlに決定。
- 2026-07-22: 表示6.594 branch-overlap版の実測は`6.693`。6.685を更新しないため不採用。
- 2026-07-22: Stage 15 fold-safe independentは`35.110`。旧CVとLBの対応仮説を棄却し、追加提出を停止。
- 2026-07-22: 5点台を必須目標とし、public strong-base + test-like branch-overlap validation +独自correctorへ計画を再編。
- 2026-07-22: 次のactive taskをStage 16A frontier差分監査に設定。
- 2026-07-22: Stage 16A完了。230/240は上流同一で、6.693版に新しいbranch-overlap modelがないことを確認。active taskをStage 16Bへ更新。
- 2026-07-22: Stage 16Bローカルfull完了。短prefix primaryでlast-known TVT RMSE 23.096、constant-U 103.911、linear-U 59.808。旧delta-U検証の分布不一致を確認し、active taskをStage 17へ更新。
