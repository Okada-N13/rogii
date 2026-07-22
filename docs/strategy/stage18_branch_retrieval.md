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
