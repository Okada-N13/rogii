# Stage 14: cross-fitted learned emission residual

Stage 13ではstandard/spatial/typewellの90/5/5 blendが全5 standard foldsで改善した一方、補正量・roughness・entropyを閾値化した固定gateは全familyで悪化した。Stage 14は「危険」の規則を手で決めず、Stage 12B expected predictionに残る小さな残差をfold-local HGBで学習する。

共通generic featureはfamily自身のbase/surface、expected correction、補正絶対値、隣接roughness、cut平均・標準偏差、well平均補正量、surface slope、suffix位置、cut fractionである。すべてpredictionとvisible inputから計算され、hidden TVTはtargetにだけ使う。spatial/typewellではこの共通schemaのみを使う。

standardでは追加でspatial/typewell予測との差、3 branch分散、Stage 12B entropyを使うstacked branchを学習する。各outer foldのモデルはそのfoldのwellを一切学習に含めない。raw residual correctionは直接採用せず、6個の固定weight/capを残りのinner foldsだけで選択する。90/5/5 blendは固定controlとして同じ表に残す。

promotionにはstandard gainとbootstrap、P90/worst-tail、spatial/typewell gain、および3 familyで安全な同一generic correction specを要求する。不合格なら後処理による改善を終了し、emission本体の表現・学習targetを改訂する。
