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

