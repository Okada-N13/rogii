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

## 実行

Colab CPU:

`notebooks/570_run_stage23a_strong_base_ncc.ipynb`

Kaggle submissionは生成しない。
