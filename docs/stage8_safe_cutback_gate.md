# Stage 8A: Safe MHA visible-prefix cutback gate

Stage 7Fはpackage predictionをravaghi branchへ40%混ぜたが、Kaggle hidden scoreは`6.997 -> 7.115`と悪化した。packageは正しく実行されていたため、公開packageの追加blendは不採用とする。

Stage 8Aは6.997のSafe MHAに既に含まれるGold visible-prefix補正を、学習井の擬似hidden suffixで直接監査する。公開modelをさらに追加する実験ではない。

各井戸について次を行う。

1. 既知prefixを55%、70%、84%地点で切る。
2. 切った後の既知TVTを擬似hidden holdoutとして使う。
3. `U = TVT + Z`をMD上のrobust 1--3次多項式で外挿する。
4. tail 80/160/320/allの候補から、3 cutbackのmedian RMSEで井戸別候補を選ぶ。
5. cutback gain、score、一貫性、baseとの差の大きさで補正をgateする。
6. conservative/balanced/aggressive profileをnested standard foldとspatial blockで選ぶ。

採用には通常5-fold、空間6-block、paired-well bootstrap、well P90、worst 10% SSE shareの全gate通過と、全fold/blockで悪化しない単一inference profileが必要である。

fleonggの公開Datasetには正規OOFがないため、Stage 8Aではfleongg/SP45の最終weightを探索しない。full-train predictionをtrain truthへ当てる評価は採用判断に使わない。

Colabでは[160_run_stage8_safe_cutback_gate.ipynb](../notebooks/160_run_stage8_safe_cutback_gate.ipynb)だけを開いて全セルを実行する。既存のStage 7 `base_oof.parquet`がGoogle Driveにあれば再利用し、なければ公開artifactから再構築する。`LIMIT_WELLS=None`の結果だけが採否判定に有効である。
