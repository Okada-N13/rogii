# Stage 35: functional residual curve

## Stage 34Aの結論

training OOFだけで選んだhard gateは、62 design-validationで`-0.1304`、standard `5/5`、
P90非悪化まで改善した。しかしbootstrap上限`+0.0151`、fraction不整合、typewell不整合で
不合格。予約120 wellsは未使用。

連続gateとhard gateの両方で新しい坑井群への補正可否判定が安定しなかったため、
Stage 30--34のrowwise TCN residual familyを終了する。

## Stage 35A

残差をrowごとに独立予測せず、suffix全体の滑らかな関数として予測する。

1. 固定500 training wells・1,370 cutsを使用する。
2. 強い`top_pf_a130` proxyとの差を101 rowsで平滑化し、正規化深度96点へ変換する。
3. standard foldごとに除外foldを使わずSVDを行い、上位8 FPCA基底へ圧縮する。
4. target-safeなcut descriptorから8係数をHGBで予測する。
5. 各standard foldを除外した5組のFPCA基底・平均曲線・係数モデルを作る。
6. training OOF再構成から、cut均等MSEを最小化する補正alphaを解析的に1つ固定する。
7. 62 design-validationでは5モデル平均係数と固定alphaだけを使用する。

descriptorはrequested fraction、suffix長、target-safe row featuresのmean/std/first/last、
baseと候補予測の差分統計だけで構成する。hidden suffix TVTはFPCA教師・係数教師・評価以外に
使わない。

## 意図

3係数trajectoryは表現力不足、rowwise TCNはtail不安定だった。FPCAはtraining residualで
観測された非線形形状を8個の滑らかな基底として保持しつつ、高周波補正を構造的に禁止する。

補正alphaはvalidationで調整しない。最大`0.75`、movement cap `8 ft`、ramp `96 rows`を固定する。

## 昇格条件

従来どおりRMSE `-0.10`以上、bootstrap上限0未満、P90非悪化、standard 4/5、
fraction 3/4、spatial/typewell/branch各4 group以上をすべて要求する。

全条件通過時のみStage 35Bで予約120 wellsを一度開封する。Kaggle submissionはまだ作らない。

## 実行

`notebooks/720_run_stage35a_functional_residual_curve.ipynb`をColabで単独実行する。
HGBとSVD中心なのでCPUで実行可能。GPUは使用しない。
