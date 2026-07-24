# Stage 30: smooth residual TCN

## Stage 29A実測

固定500 wellsの1,370 safe cutsへ拡張し、固定weight 0.20は
`8.9038 -> 8.8284`（`-0.0754`）だった。P90、fraction、spatialは通過し、
bootstrap上限も`+0.0000677`まで縮小したが、事前基準`-0.10`、typewell、branchは不合格。

診断weight 0.30は`-0.1017`だったが、観測後にprimaryを変更しない。予約120 wellsは未使用。
HGB rowwise familyは終了する。

## Stage 30A

Stage 29Aと同じ1,370 cuts、27 target-safe features、101-row平滑残差を使う。変更点は
各rowを独立に回帰するHGBを、dilated temporal convolutionへ置き換えることだけである。

- training stride: 8
- TCN: 48 channels、5 residual blocks、kernel 5
- chunk: 256 rows、64-row overlap
- loss: Smooth L1
- 5 standard-fold models
- 各モデルのepoch選択は除外foldのscaled residual RMSE
- design validationは学習・early stoppingに使わない

validationでは5 model平均を全suffix行へ適用する。固定primaryはweight 0.20、cap 8 ft、
96-row rampのまま。Stage 29Aと同じ全gateを通過した場合だけ、予約120 wellsを一度開封する。

Stage 30AはGPU学習。L4またはA100推奨、T4でも実行可能。Kaggle提出は生成しない。

