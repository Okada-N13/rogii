# Stage 28: expanded strong-base residual field

## Stage 27Aの結論

62 design-validation cutsで、prefix XY平面が予測する終点U変化と真値の相関は
`0.923--0.927`だった。しかし強いA130 proxyへ混ぜると固定primaryは
`8.9038 -> 10.0676`（`+1.1638`）、bootstrap 95%は`[+0.6004,+1.6091]`、
standard `0/5`、branch `0/5`、fraction `0/4`で全面的に悪化した。

固定grid oracleは`-0.6394`なので一部cutには余地があるが、低周波の地層傾斜は強いbaseが
すでに取り込んでいる。XY平面を無条件に足す方向は終了する。予約120 wellsは未使用。

## Stage 28A

Stage 22のrowwise HGBは63 training wells前後しか使っていなかった。Stage 28AではStage 24の
固定splitから500 training wellsを使い、58-well design validationとはwellを完全分離する。
予約120 confirmation wellsは読み込まない。

モデルは5個のHGB ensembleで、各モデルはstandard foldを1つ除外して学習する。教師は
強い`top_pf_a130` proxyのrow residualを101 rowsで平滑化し、±24 ftへclipした値である。
raw高周波誤差を追わず、baseが外した低周波残差だけを学習する。

Stage 22の特徴に次を追加する。

- prefix-only XY planeとbaseの差
- suffixのX/Y/水平距離
- baseのU勾配
- prefix XY planeのgx/gy

すべて実testで利用可能なMD/X/Y/Z/GR/TVT_input、typewell、base候補だけから作る。
hidden suffix TVTは教師と評価以外で読まない。

固定primaryはweight `0.20`、cap `8 ft`、96-row ramp。`0.10/0.30`は診断専用。
Stage 28Bへ進むには、design validationでRMSE `-0.10`以上、bootstrap上限0未満、
P90非悪化、standard/typewell/branch 4 groups、spatial 4 groups、fraction 3 groups以上を
すべて通す必要がある。

全gate通過時のみ、事前固定した同一profileを予約120 wellsへ一度だけ適用する。

