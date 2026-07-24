# Stage 34: OOF-selected conservative TCN gate

## Stage 33Aの結論

Stage 33Aはpooled `-0.1153`、standard `4/5`、spatial `5/6`、branch `4/5`、
fraction gate通過まで改善した。しかしbootstrap上限`+0.00850`、cut P90 `+0.1394`、
typewell `3/5`で不合格。fold 0も`+0.0894`悪化した。

training cross-fit gate MAEは`0.4248`、validation平均gateは予測`0.6513`対oracle
`0.5741`で、連続gateが補正を適用しすぎている。予約120 wellsは未使用。

## Stage 34A

Stage 33Aの保存済みtraining/validation cut reportだけを使うCPU監査である。
TCN推論、特徴再生成、モデル再学習は行わない。

training OOFの`crossfit_gate`について、事前固定したscore quantile
`50/60/70/80/90/95%`をhard-gate候補とする。各候補は次の挙動になる。

- scoreが閾値以上: Stage 33のfull correctionを使用
- scoreが閾値未満: baseをそのまま使用

閾値はtraining OOFだけで選ぶ。OOF RMSE `-0.05`以上、bootstrap上限0未満、
P90非悪化、standard 4/5、fraction 3/4、active 10%以上をすべて通過した候補のうち、
OOF RMSE最良を固定する。

固定閾値を62 design-validation cutsへ一度適用し、従来の全gateで判定する。
validation真値を閾値選択には使わない。Stage 34Aでも予約120 wellsは開封しない。

Stage 34A不合格ならTCN residual familyを終了し、追加weight・閾値探索を行わない。
