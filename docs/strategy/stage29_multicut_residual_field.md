# Stage 29: multi-cut residual field

## Stage 28A実測

500 training wells、297,725 sampled rows、5-model HGB ensembleで、固定weight 0.20は
design validationを`8.9038 -> 8.8342`（`-0.0696`）へ改善した。standardとbranchは
各`4/5`改善したが、bootstrap上限`+0.0299`、cut P90 `+0.0728`、spatial/typewell
各`3 groups`、fraction gate不合格でStage 28Bへ進まない。

weight 0.30は`-0.0881`だったが、観測済みvalidationを根拠にprimary weightを変更しない。
予約120 wellsは未使用。

## Stage 29A

モデル、特徴、weight、cap、validationを変更せず、固定500 training wellsについて
`evaluation_role=primary`かつ`replay_eligible=True`の全short-prefix cutsを学習へ使う。
井戸数を水増しせず、同一井戸内の異なるprefix長を教師へ追加してfraction generalizationを
改善する実験である。

学習row数をStage 28Aと同程度に保つためstrideを8から32へ変更する。教師は同じ101-row
平滑残差、モデルは同じ5-fold HGB ensemble、primaryはweight 0.20のままである。

62 design-validationのgateもStage 28Aと同一。全gate通過時だけ予約120 wellsを一度開封する。
不通過ならHGB rowwise residual familyを終了する。

