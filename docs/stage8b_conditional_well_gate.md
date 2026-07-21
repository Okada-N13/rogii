# Stage 8B: conditional per-well physics gate

Stage 8Aではconservative profileが`10.370056 -> 10.370323`でほぼ同点だったが、全inner foldを悪化させないprofileはなく不採用となった。BalancedとAggressiveは明確に悪化した。

Stage 8Bはconservative補正を一律適用せず、prefixから計算可能な診断値だけを用いて適用井戸を選ぶ。

井戸別gateの入力は以下である。

- cutback best/anchor RMSEとgain
- 3 cut地点での候補一貫性
- 選択された多項式degreeとtail
- known/hidden行数
- base軌道のrange、std、slope
- physicsとbaseの差のmean/std/RMSE/P95/max/slope
- conservative補正の移動量
- hidden区間のMD span

真のhidden TVTから計算した改善量は学習targetにだけ使用し、入力特徴には含めない。通常foldと空間blockの各outer groupについて、残りgroup内でさらにinner OOF predictionを作り、全inner groupを悪化させないthresholdだけをouter groupへ適用する。

最終inference thresholdは、通常5-fold cross-fitと空間6-block cross-fitの双方で全groupを悪化させない候補から選ぶ。全gate通過かつthresholdが非nullの場合だけKaggle移植へ進む。

[170_run_stage8b_conditional_well_gate.ipynb](../notebooks/170_run_stage8b_conditional_well_gate.ipynb)をColabで全セル実行する。Stage 8A v002のcandidate matrixがあれば再利用する。
