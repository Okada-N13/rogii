# Stage 31: TCN ensemble uncertainty shrinkage

## Stage 30A実測

1,370 cuts、875,227 sampled rowsのTCNは、固定weight 0.20で
`8.9038 -> 8.8102`（`-0.0936`）だった。standard/spatial/branch/fractionは通過したが、
bootstrap上限`+0.0160`、P90`+0.2877`、typewell `3/5`、事前基準`-0.10`で不合格。
予約120 wellsは未使用。

診断weight 0.30は`-0.1334`だが、そのまま後付け採用しない。

## Stage 31A

Stage 30Aの5 checkpointsを固定し、再学習せず62 design-validation cutsだけを再推論する。
各rowの5予測からensemble meanとspreadを求める。

固定primary:

```text
confidence = |mean| / (|mean| + std)
residual = mean * confidence
weight = 0.30
```

weight 0.30の強い平均信号を使いつつ、モデル間で不一致のrowを自動的に縮小する。
confidence power 0.5/2.0とsign agreementは診断専用。hidden suffix TVTはgate計算に使わない。

Stage 30Aと同じRMSE、bootstrap、P90、全group gateをすべて通過した場合だけ、
この固定primaryを予約120 wellsへ一度だけ適用する。

## 実測結果

固定primary `confidence_w030_p100` は `8.903811 -> 8.803360`
（`-0.100450`）まで改善した。全group consistencyは通過したが、
bootstrap 95%上限が`+0.00959`、cut P90 deltaが`+0.25152`だったため不合格。
予約120 wellsは未使用。

同じ実行で事前定義していた診断候補は次の順だった。

- `agreement_w030_a060`: `-0.145046`
- `confidence_w030_p050`: `-0.116532`
- `confidence_w030_p100`: `-0.100450`
- `confidence_w030_p200`: `-0.073191`

主候補だけでTCNを終了せず、これら4候補を保存済みcut report上で同一gateにかける
Stage 32Aを行う。追加学習、再推論、閾値追加、予約データの利用は行わない。
