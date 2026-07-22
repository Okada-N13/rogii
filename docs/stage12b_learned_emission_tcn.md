# Stage 12B: 61状態 learned-emission TCN

Stage 12Aでは、raw NCCの正解候補 Top-10 recall が26.1%（ランダム16.4%）で、正解状態を候補集合へ残す信号は確認できた。一方、raw argminのRMSEは47.1であり、反復するGR模様だけではbranchを一意に選べない。Stage 12Bはこの「候補生成には使えるが、単独選択には使えない」という結果を受け、61個のoffset候補の確率分布を学習する。

## 固定するもの

- surface prior: Stage 11Cの `w075_cap50`
- offset grid: -60〜+60 ft、2 ft刻みの61状態
- 入力: `ncc_w5`, `ncc_w13`, `ncc_w25`, `ncc_mix`、水平井GR、各候補のtypewell GR、候補有効mask
- OOF: Stage 11と同じwell単位5-fold
- 正解TVTの使用箇所: nearest stateラベルと評価だけ

各候補状態には同じTCNを適用する。固定NCCに加えて、水平井GRと候補typewell GRを対にしたチャネルを共有TCNへ渡すため、単なるNCC重み付けではなくGR形状の対応も学習できる。さらにoffset値、suffix内位置、cut fraction、surface slopeを使う。候補間で重みを共有するため、特定offset番号を丸暗記しにくい。cross entropyに加え、raw `ncc_w5`が選んだ誤候補をhard negativeとしてmargin lossへ加える。

## 判定

Stage 12Cの空間・typewell cross-fitとK-best pathへ進む条件は次の通り。

- learned Top-10がraw NCCより2ポイント以上改善
- learned Top-5がraw NCCより改善
- 5 fold中4 fold以上でTop-10改善
- learned NLLが温度固定raw NCCより改善
- hidden-target invarianceが維持される

expected-offset RMSEとMAP RMSEは表示するが、ここではpromotion gateにしない。独立なrow argmaxはStage 12Aと同じbranch ambiguityを残すため、最終判断には連続性制約を持つdeterministic K-best latticeが必要になる。

Colabでは [280_run_stage12b_learned_emission_tcn.ipynb](../notebooks/280_run_stage12b_learned_emission_tcn.ipynb) をT4で実行する。`LIMIT_WELLS = None`が本判定であり、Kaggle提出ファイルは生成しない。
