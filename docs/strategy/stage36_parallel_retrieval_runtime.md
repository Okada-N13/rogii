# Stage 36: exact-parity parallel retrieval runtime

## 判断

Stage 35Aのfunctional residual curveはtraining OOFで改善したが、独立62-cut検証では
`-0.0924`、bootstrap上限`+0.2147`、P90`+0.3720`、standard `2/5`となり棄却する。
予約120 wellsは開封しない。cut単位の残差曲線学習はここで終了する。

次は、未提出のまま残っているStage 18 learned donor retrievalをKaggle時間制限内へ収める。
Stage 18Cは全3,865 cutsで`-1.740`、5/5 folds・全fraction・P90・bootstrapを通過し、
Stage 18Dのfold-safe rankerも固定retrievalからさらに`-0.139`改善した。これまでの独自手法で
最も強い再現信号であり、失敗理由は精度ではなくsubmissionの時間超過だった。

## 変更するもの

- test wellを最大4本同時に処理する。
- 読み取り専用のpacked donor trajectory、global KD-tree、portable rankerを共有する。
- donorごとのKD-treeをlock付きで共有キャッシュする。
- 各threadはwellの予測配列とauditだけを返す。
- 最終DataFrameへの代入はwell ID順に単一threadで行う。

donor候補、fold assignment、特徴量、ranker、選択donor、距離重み、blend weight `0.20`は
一切変更しない。

## 昇格条件

`notebooks/730_run_stage36_parallel_retrieval_benchmark.ipynb`をColab CPUで単独実行する。

1. 1 workerと4 workersのsubmission CSVがbyte単位で一致する。
2. 両方ともsample ID順、finite TVT、全public wells auditを通過する。
3. hidden 200 wellsの保守的推定時間が600秒以下である。

推定式は、観測したparallel public時間を固定費としてそのまま残し、残りのwellを
`serial_seconds / public_wells`のコストで4並列batch処理する。小規模benchmarkで固定費を
過小評価しないための保守的な式である。

全条件を通過した場合だけpackage v4をKaggle Datasetへアップロードし、notebook 460を実行する。
GPUはStage 18後処理には不要で、公開base notebook側の推奨acceleratorを維持する。
