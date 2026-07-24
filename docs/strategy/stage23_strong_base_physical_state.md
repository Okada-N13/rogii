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

## Stage 23B実測結果

rankerは外部validationで明確に成功した。top10 `30.70%→70.33%`、top5
`18.60%→49.98%`、median rank `20→6`、NLL `-1.2057`。standard 5/5、
spatial 6/6、typewell 5/5、branch 5/5、fraction 4/4の全groupでtop10が改善した。

一方、posterior expected offsetを固定weight 0.50で直接使うdecoderはRMSE
`8.6132→8.5244`（`-0.0889`）に留まり、事前閾値`-0.10`未達、bootstrap上限
`+0.0345`、well P90 `+0.2636`で不合格だった。weight 0.25/0.75/1.00も診断上は
すべて平均改善したが、validation結果から後付け採用しない。

Stage 23CではTCNを変更せず、Stage 21A training OOF logitsだけでdecoderをnested
cross-fitする。direct posterior、affine ridge、posterior summary ridgeを事前固定比較し、
training OOF上でgain、bootstrap、fold、P90をすべて通過したprofileだけを全training OOFで
refitする。そのprofileを固定してからStage 21B validationへ適用する。

## Stage 23C実測結果

training OOFでは`summary_a1000`が最良で、RMSE `-0.1408`、well P90 `-1.2421`だった。
しかし改善は4/5 foldsに留まり、worst foldは`+0.2387`、bootstrap 95%上限は
`+0.1968`だった。ほかのaffine/summary profileもworst foldが`+0.3317`から
`+0.6570`で、eligible profileは0件だった。規則どおりStage 21Bへ補正を適用せず、
validation deltaは`0.0`となった。

結論は、学習済みTCNのrank信号は維持する一方、posteriorからrowwise連続offsetを直接
回帰するdecoder familyを棄却する、である。

## Stage 23D

Stage 23Cが保存したtraining OOF/validation posterior特徴を再利用する。TCN推論やNCC volume
生成は行わない。decoderを次の3要素に分解する。

1. `|offset| >= 3 ft`となる移動確率
2. 移動する場合の正負方向確率
3. 移動する場合の絶対offset量

2つの確率は強く正則化したlogistic regression、絶対量はridgeで学習し、確率積によって
補正を自動縮小する。4つのprofileは実行前に固定し、Stage 21A training OOF内のnested
cross-fitでのみ選ぶ。Stage 21BはStage 23B/Cの結果をすでに観測しているため、もはや
完全未使用holdoutとは呼ばずdesign validationとして扱う。Stage 23Dが通っても提出せず、
新しいdisjoint well確認を必須とする。

## Stage 23D実測結果

階層化によってStage 23Cよりfold安定性は改善した。最良`h_c003_a1000_w075`はtraining
OOFでRMSE `-0.1071`、P90 `-0.6347`、4/5 folds改善だった。しかしworst foldは
`+0.0817`、bootstrap上限は`+0.0461`で不合格。より強く正則化した
`h_c001_a1000_w075`はworst fold `+0.0498`まで満たしたがbootstrap上限`+0.0482`で、
eligible profileは0件だった。したがってdesign validationへ補正を適用していない。

この結果から、decoderのweight/C/alpha調整を続けない。主要な制約は77 training cutsという
標本数と、one-hot state分類がexpected offsetのRMSE校正を直接最適化しない点である。

## Stage 24A

Stage 21B design-validation 58 wellsを除外し、stage16 standard foldごと100 wells、
合計500 wellsから1 cutずつを決定論的hashで選ぶ。さらにfoldごと24 wells、合計120 wellsを
Stage 24B確認用として予約し、Stage 24Aでは特徴生成・学習・評価のいずれにも使わない。

lossはone-hot cross entropyから次へ変更する。

- true offset近傍にも確率を与えるGaussian soft ordinal target（sigma 2 ft）
- posterior expected offsetとtrue grid offsetのHuber loss
- hard-negative marginは補助lossとしてweightを半減
- early stoppingはNLLでなくinternal-fold expected-offset RMSE

primary correction weightは実行前固定`0.75`。Stage 21Bはdesign validationであり、
通過してもKaggle packageを作らず、予約120 wellsのStage 24Bへ進む。

初回manifest v001は`replay_eligible=False` cutを含み、public OOF開始位置より前のcutで
prediction length mismatchとなった。モデル計算前に停止しており結果は生成されていない。
v002では`evaluation_role=primary`かつ`replay_eligible=True`を必須にし、loaderでも
public OOFがwell末尾まで連続すること、cutがOOF開始以降であること、suffix長が一致することを
再検査する。

v002 manifest生成は詳細stderrをNotebookへ表示しないまま停止した。一律
`100 training + 24 confirmation`を各foldへ要求する設計ではeligible well数のfold差を
許容できないため、これも同時に修正する。v003では
confirmation 24/foldを維持し、training総数500を各foldの余剰capacityへ決定論的water-fill
再配分する。manifest subprocessのstdout/stderrは常に表示する。

## 実行

Colab CPU:

`notebooks/570_run_stage23a_strong_base_ncc.ipynb`

Stage 23B（T4 GPU）:

`notebooks/580_run_stage23b_learned_emission.ipynb`

Stage 23C（T4/L4 GPU、TCN再学習なし）:

`notebooks/590_run_stage23c_oof_decoder.ipynb`

Stage 23D（CPU、Stage 23C artifacts再利用）:

`notebooks/600_run_stage23d_hierarchical_decoder.ipynb`

Stage 24A（A100/L4推奨、T4可）:

`notebooks/610_run_stage24a_scaled_ordinal_emission.ipynb`

Kaggle submissionは生成しない。
