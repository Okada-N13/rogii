# Stage 33: fold-safe TCN cut-benefit gate

## 背景

Stage 32Aでは事前定義済み4 profileを完全監査したが、全候補でbootstrap上限が正、
cut P90も悪化した。最良`agreement_w030_a060`はpooled `-0.1450`、standard `4/5`、
fraction `4/4`まで改善しており、TCN補正方向そのものには信号がある。一方で悪化cutを
rowwise ensemble uncertaintyだけでは識別できなかった。

予約120 wellsは未使用のままTCN uncertainty profile familyを終了する。

## Stage 33A

既存Stage 30Aの5 TCN checkpointsを固定する。追加のTCN学習やweight探索は行わない。

1. 固定500 training wells・1,370 cutsを使う。
2. training cutには、そのcutのstandard foldを学習していないcheckpointだけを適用する。
3. 固定weight `0.30`、cap `8 ft`、ramp `96 rows`でfull correctionを作る。
4. baseとfull correction間のSSE最適係数を`0..1`で解析的に求める。
5. target-freeなcut要約特徴から、この係数を5-fold HGBで学習する。
6. 62 design-validation cutsでは5 TCN平均と5 gate-model平均を使う。
7. gate係数以外のTCN出力、weight、cap、rampは変えない。

gate特徴はrequested fraction、suffix長、TCN residual統計、normalized row-feature統計、
baseと各target-safe candidateの差分統計だけで構成する。hidden suffix TVTは教師と評価以外に
使用しない。

## 昇格条件

- pooled RMSE delta `<= -0.10`
- well bootstrap 95%上限 `< 0`
- cut P90非悪化
- standard `4/5`以上
- fraction `3/4`以上
- spatial/typewell/branch各4 group以上
- training/design-validation well overlapゼロ
- hidden-target invariance

全条件を通過した場合のみ、モデルと設定を凍結してStage 33Bで予約120 wellsを一度だけ開封する。
Stage 33AからKaggle submissionは作らない。

## 実行環境

`notebooks/700_run_stage33a_tcn_cut_benefit_gate.ipynb`をColabで単独実行する。
TCN再学習はないが1,370 cutsのfold-safe推論を行うためGPUを使う。L4/A100推奨、T4でも可。
