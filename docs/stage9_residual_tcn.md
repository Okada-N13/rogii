# Stage 9A: independent residual TCN

Stage 8A/8Bではvisible-prefix physics補正と井戸別gateの双方が不採用になった。Stage 9Aは公開packageを追加せず、ravaghi OOF baseの残差を独自TCNで学習する。

入力はcompetition dataと固定base predictionから再構築する21特徴である。

- base delta、surface delta、slope、curvature
- MD since prefix、eval fraction
- Z/XY deltaとMD微分
- GRのprefix正規化、rolling 9/31、gradient
- base TVTでtypewell GRを参照した残差とgradient
- Stage 8で生成済みのphysics/base disagreementとconservative move

hidden TVTはresidual targetにだけ使用し、特徴生成には使用しない。不変性テストではhidden TVTを変更しても全特徴が同一であることを確認する。

初期TCNは32 channels、4 residual blocks、kernel 5、dilation 1/2/4/8、dropout 0.15である。学習時だけ4行strideを使い、検証推論は全行で行う。5-fold GroupKFoldごとにfold外の井戸だけでfeature standardizerとTCNを学習する。

補正weightは`0.05/0.10/0.20/0.35/0.50/0.75/1.0`からnested OOFで選ぶ。各outer foldでは残りinner foldを一つも悪化させないweightだけを適用する。

Stage 9Aは通常foldの信号確認であり、Kaggle提出を許可しない。全standard gateを通過し、全foldで悪化しないinference weightが得られた場合のみStage 9Bの空間6-block cross-fitへ進む。

[180_run_stage9_residual_tcn.ipynb](../notebooks/180_run_stage9_residual_tcn.ipynb)をT4またはP100のColabで全セル実行する。mixed precisionを使うためT4を優先する。
