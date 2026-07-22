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
