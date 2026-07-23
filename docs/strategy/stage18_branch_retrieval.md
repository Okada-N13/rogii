# Stage 18: target-free branch retrieval

## Stage 18A目的

Stage 17 strong baseとは異なる信号として、近接する別well trajectoryから地層面`U = TVT + Z`を取得する。疑似test wellと同じfoldのdonor targetは使用しない。

## 予測方法

1. Stage 16Bのtarget-free donor graphから上位4 donor wellsを取る。
2. standard/spatial/branch-group familyごとに同じfoldのdonorを除外する。
3. target trajectory各点からdonor trajectoryの最近傍XYZ点を求める。
4. donorの`U`を取得する。
5. visible prefix末尾256 rowsでtarget Uとの差のmedianを取り、donor Uをcalibrateする。
6. XYZ距離とGR差で重み付け平均する。
7. Stage 17 strong baseへ20%だけblendする。

## 固定ablation

- `geometry_w020`: calibrationなし、GRなし
- `prefix_geometry_w020`: visible-prefix calibrationあり、GRなし
- `prefix_gr_w020`: calibration + GR、20%（primary）
- `prefix_gr_w035`: calibration + GR、35%（診断のみ）

## 検証

- primary 5 folds × 5 fractionsから各6 cuts、合計150 cutsをSHA-256固定抽出
- standard donor fold exclusion
- spatial donor block exclusion
- branch-group fold exclusion
- well bootstrap
- cut P90/max

primary profileは結果を見る前に`prefix_gr_w020`へ固定する。通過後だけall-cut化とlearned donor rankingへ進む。

## 実行

`notebooks/410_run_stage18a_branch_retrieval.ipynb`をCPU Colabで実行する。Kaggle提出は行わない。

## Stage 18A実測と判断

固定150 cutsで次を得た。

- standard primary: `18.565 → 16.356`（`-2.209`）、4/5 folds。ただしcut P90は`+1.645`、well bootstrap 95%は`[-1.804, +0.309]`。
- branch-group primary: `18.565 → 15.850`（`-2.715`）、5/5 folds、cut P90 `-0.346`。
- branch-group 35%診断: `-3.873`だが、結果確認後の強い係数なので採用しない。
- spatial primary: donorが2本以上残ったのは32/150 cutsだけで、`+3.707`悪化。
- calibrationなしは全familyで大幅悪化。visible-prefix calibrationは必須。
- GRの追加効果は小さく、主信号は近接trajectoryとprefix offsetである。

Stage 18A全体はpromoteしない。spatial除外は近接donorをほぼ消して遠方外挿へ変えるため、今回のlocal branch retrievalと同じ推論条件を測っていない。一方、branch-group除外はfold一貫性とtailの両方を満たしたため、独立holdoutで一度だけ確認する。

## Stage 18B: 独立branch-group確認

- Stage 18Aのhash順位0–5を再利用せず、各stratumの順位6–11から150 cutsを固定する。
- familyは`branch`だけ、profileは事前固定した`prefix_gr_w020`だけにする。
- blend weightは20%のまま。35%や閾値を探索しない。
- 合格条件はRMSE `-0.30`以上、4/5 folds以上、cut P90非悪化、well bootstrap上限`< 0`。
- 合格時だけ全primary cutsへ拡張し、その後learned donor rankingへ進む。
- 不合格なら固定retrievalを提出系へ入れず、donor ranking自体を学習化する。

実行Notebookは`notebooks/420_run_stage18b_branch_confirmation.ipynb`。CPU Colabでよく、Kaggle提出は行わない。

## Stage 18B実測と判断

Stage 18Aと重複0の独立150 cuts（sample SHA-256 `63f67a034ccc2676a88ea49a54d3b4ece3f3f7c0aa660a751d1d0a18e8151fd6`）で全gateを通過した。

- RMSE: `12.010 → 10.399`（`-1.611`）
- fold: 5/5改善。最小改善のfold 3でも`-0.768`
- cut P90: `-2.223`
- cut max: `-8.509`
- well bootstrap 95%: `[-1.317, -0.339]`

Stage 18A branch-group結果と方向が一致し、別sampleでもtailを含めて改善した。固定retrieval信号はall-cut検証へ昇格する。

## Stage 18C: 全primary-cut検証

- Stage 16Bの全primary cuts（想定3,865）を対象にする。
- familyはbranch-group除外、profileは`prefix_gr_w020`だけ。
- fold 5/5に加え、5つのprefix fractionがすべて改善することを要求する。
- cut coverage 99%以上、cut P90非悪化、well bootstrap上限`< 0`を要求する。
- 同一well–donorの最近傍対応をcacheし、全cut処理を高速化する。
- 通過後はこの全cut結果をcontrolとして凍結し、learned donor rankingを評価する。

実行Notebookは`notebooks/430_run_stage18c_all_cut_branch_retrieval.ipynb`。CPU Colabで実行し、まだKaggle提出は行わない。

## Stage 18C実測と判断

全3,865 primary cuts、18,841,328 suffix rowsで全gateを通過した。

- RMSE: `14.524 → 12.784`（`-1.740`）
- coverage: 100%
- fold: 5/5改善（`-1.080`から`-2.861`）
- prefix fraction: 5/5改善
  - 0.18: `-3.565`
  - 0.22: `-1.532`
  - 0.26: `-0.909`
  - 0.30: `-0.593`
  - 0.34: `-0.578`
- cut P90: `-1.229`
- cut max: `-68.614`
- well bootstrap 95%: `[-1.125, -0.680]`
- sample SHA-256: `f05e478d8c152f0b1f99179028c44569ebbac918756f52d7af3fe7c5c4b55632`

固定top-4 branch retrievalを次のcontrolとして凍結する。短prefixだけでなく全fractionで改善しているため、retrieval信号自体は再現性がある。

## Stage 18D: cross-fitted learned donor ranking

Stage 18Cはbranch-group外候補を元のgeometry順位から上位4本選んでいる。Stage 18Dでは最大12候補から、target-free特徴だけで4本を選ぶ。

特徴:

- donor graphのXYZ距離、prefix matched points、GR差、typewell GR差
- visible prefix上のcalibrated U誤差・相関・offset
- 公開される全trajectory上のXYZ距離とGR差
- donor Uのsuffix形状とStage 17 strong baseとの差
- cut fraction、prefix/suffix長

fold安全性:

- 評価branch-group foldをtarget roleだけでなくdonor roleからも学習除外する。
- suffix TVTはdonor品質labelにだけ使用し、推論特徴には入れない。
- 5-fold OOF scoreでdonorを選び、Stage 18C固定top-4と直接比較する。
- model/hyperparameters、候補12本、選択4本、blend 20%を事前固定する。

合格条件は固定control比`-0.05`以上、4/5 folds、4/5 fractions、P90非悪化、well bootstrap上限`< 0`、coverage 99%以上。通過時だけ全data rankerとindependent test inferenceへ進む。

実行Notebookは`notebooks/440_run_stage18d_learned_donor_ranker.ipynb`。CPU Colabで実行し、Kaggle提出は行わない。

## Stage 18D実測と判断

43,758 candidate rows、全3,865 cutsで全gateを通過した。

- OOF score Spearman: `0.9625`
- oracle top-1 recall: `0.5915`
- 固定top-4との平均donor overlap: `0.5463`
- Stage 18C固定retrieval比: `12.784 → 12.645`（`-0.139`）
- standard folds: 5/5改善
- branch-group folds: 4/5改善。fold 3のみ`+0.0099`
- prefix fractions: 5/5改善
- cut P90: `-0.243`
- cut max: `-0.545`
- well bootstrap 95%: `[-0.205, -0.0099]`
- strong base比: `-1.879`

改善幅は小さいが、fold/fraction/tail/有意性の事前gateをすべて満たしたためrankerを昇格する。LB係数探索は行わない。

## Stage 18E: fold-safe test inference package

実testへcross-fit条件を維持するため、単一full-data modelではなく5個のfold-safe modelをpackage化する。

- test well IDがtrain assignmentsにも存在する場合、その凍結branch-group foldを使う。
- 評価foldモデルは、学習時にそのfoldをtarget roleとdonor roleの両方から除外済み。
- 同じwell IDのtrain TVTをtest donorとして使うことを明示的に禁止する。
- test IDがtrainにない場合だけvisible-prefix branch vote、さらに該当なしならstable hash foldを使う。
- final 6.685 V599 submissionへ、選択4 donorのretrievalを20%適用する。
- inferenceはInternet-OFF、competition train/testとpackageだけで完結する。

package構築Notebookは`notebooks/450_build_stage18e_ranked_retrieval_package.ipynb`。推奨Kaggle Dataset名は`rogii-stage18e-ranked-retrieval-package`。

## Stage 18E実測とStage 18F提出候補

Stage 18E packageは正常に完成した。

- fold-safe ranker: 5 models
- training candidate rows: 43,758
- same-well target leakage guard: 有効
- package manifest SHA-256: `7bddc1914f3d046b678dbb8f5d1cc17427b03bc85c1a06d1f2088cbe68d3935d`
- archive: `stage18e_ranked_retrieval_package.zip`

Kaggle用Notebookは`notebooks/460_kaggle_v599_stage18_ranked_retrieval.ipynb`。これは実測6.685の`230_kaggle_v599_a130_frontier_safe.ipynb`を固定し、既存branch hedgeの後・final auditの前にだけStage 18 retrievalを追加する。

実行条件:

- Internet OFF
- AcceleratorはT4 x2（既存V599推論に合わせる）
- 既存V599 Notebookで使用した全Datasetに加え、推奨名`rogii-stage18e-ranked-retrieval-package`を追加
- DatasetはZIPのままでも展開済みでも検出可能
- 3 test wellsすべてで`status=applied`にならなければhard fail
- `STAGE18E_TEST_AUDIT`、最終`V599_FRONTIER_SAFE_AUDIT`、`submission.csv`だけを採用する

これはStage 18Fの初回提出候補であり、5点台を保証するものではない。疑似testで確認した改善が実testでも再現するかを、6.685 controlに対する単一の事前固定比較として評価する。LB結果が悪化した場合、weight変更の連続提出はせず、auditと分布差を診断する。

## Stage 18F初回実行の時間超過とv002

v001はinteractive auditを正常完了したが、Kaggle submission rerunが時間内に完了せずスコアが付かなかった。精度gateの失敗ではなくruntime failureとして扱う。

原因候補は、上限近くまで使うV599本体の後でStage 18が773 horizontal CSVと773 typewell CSVを再読込していたこと。v002 packageでは全donor trajectoryとtypewell GR平均を単一`donor_trajectories.npz`へ事前packする。推論式、fold assignment、ranker、候補選択、20% blendは変更しない。

- package build run: `stage18e_ranked_retrieval_package_v002`
- 同じKaggle Datasetを新versionで置換する
- Kaggle Notebookはv002 cacheがなければhard failする
- `STAGE18E_TEST_AUDIT`の`donor_source=packed_npz`と`elapsed_seconds`を確認する

## Stage 18F hidden-test監査修正（v003）

v002のpublic placeholder 3 wellsではStage 18が`26.30秒`で完了し、packed cacheが正常に使われた。したがってpublic実行時間が大きく変わらないのは正常で、V599本体が支配的である。

一方、このコンペのsubmission rerunでは3 fake wellsが約200 real wellsへ置換される。v002 Notebookに残っていた`len(statuses) == 3`監査はhidden rerunで必ず失敗するため修正した。

v003では:

- sample submissionから実際のwell数を求め、3/200の両方を監査する
- donor不足だけは安全なbase fallbackとして許可する
- 同一donorのKD-treeをtest wells間で再利用する
- package manifestに`package_version: 3`を記録し、古いpackageを拒否する

3-well所要時間からStage 18単体の200-well単純推定は約29分。KD-tree再利用により実際はこれ以下を狙う。V599本体はpublic表示時間ではなく200-well hidden scalingで管理する。
