# Stage 23: strong-base aligned physical state

## 背景

Stage 19/20の3係数trajectory residual、Stage 21のvisible-prefix candidate router、
Stage 22のrowwise candidate-disagreement residual fieldはいずれもdisjoint wellで安定しなかった。
共通の問題は、強いA130 baseの誤差を直接回帰または候補選択しようとした点にある。

Stage 12ではraw NCC自体の予測値は悪かった一方、真のoffset stateに対するtop-k rank信号と
大きなoracle headroomが観測された。Stage 23ではこの物理状態を現在の強いA130 proxyへ
直接合わせ直す。

## Stage 23A

Stage 21Bの固定62 cuts・58 wellsだけをvalidationに使う。A130 baseの各rowについて、
`-30..+30 ft`を1 ft刻みにした61個のTVT offset stateを置く。各stateでtypewell GRを補間し、
horizontal GRとのrolling NCC costをwindow 5/13/25で計算する。primary emissionは
window 13/25の`0.4/0.6` mixとする。

評価targetはstate rank、oracle、decoder RMSEの算出にだけ使う。emission、A130 base、
smooth decoderはouter suffix TVTを読まない。hidden-target invarianceを直接検査する。

固定decoder profileはweak/medium/strongの3つを診断するが、Stage 23Bへの主gateは
raw emissionのrank信号である。raw emissionは絶対予測として弱くても、learned emissionの
入力として有用な可能性があるためである。

## Gate

- hidden-target invariance
- offset grid coverage `>= 0.95`
- oracle gain `>= 3.0 RMSE`
- top10 recallがrandomの`1.25x`以上
- median true-state rank `<= 25`
- standard 4/5、spatial 4/6、typewell 4/5、branch 4/5 groupsでrandom超過
- 3/4以上のprefix fractionsでrandom超過

全gate通過時だけ、別training splitを使うStage 23B strong-base-aligned learned emissionへ進む。
不通過ならGR offset state自体を棄却し、非GRの物理状態へ移る。

## Stage 23A実測結果

2026-07-24に62 cuts・58 wells・31,744 sampled rowsで全gateを通過した。offset coverage
`99.31%`、base RMSE `8.6132`に対するoracleは`2.2731`（`-6.3401`）。
raw top10 recallは`31.14%`でrandom `16.39%`の約1.90倍、median rankは20だった。
standard 5/5、spatial 6/6、typewell 5/5、branch 5/5、fraction 4/4の全groupで
random top10を上回った。

固定raw decoderはweak `+0.0963`、medium `+0.1568`、strong `+0.0234`でまだ絶対予測を
改善しない。これはStage 12Aと同様に、raw costのargmin/単純posteriorは弱いがrank情報は
学習可能、という結果である。profile調整は行わずStage 23Bへ進む。

## Stage 23B

Stage 21Aの77 cuts・63 wellsだけをtraining、Stage 21Bの62 cuts・58 wellsだけを固定外部
validationにする。training内のstage16 5-foldごとにTCNを学習し、validationでは5 modelの
logit平均を使う。validation targetはearly stopping、epoch選択、model選択に使わない。

入力は4 NCC channels、horizontal/typewell GR、candidate validityの7 state-shared channelsと、
suffix progress、base slope、GR、selector/public disagreementのrow featuresである。
primary correctionはposterior expected offsetのweight `0.50`、cap `12 ft`、ramp `64 rows`に
結果を見る前から固定する。

rank改善だけでなく、固定primaryのRMSE gain、bootstrap上限、well P90、全fold familyと
fractionでのrank consistencyをすべて通過した場合だけStage 23Cへ進む。

## 実行

Colab CPU:

`notebooks/570_run_stage23a_strong_base_ncc.ipynb`

Stage 23B（T4 GPU）:

`notebooks/580_run_stage23b_learned_emission.ipynb`

Kaggle submissionは生成しない。
