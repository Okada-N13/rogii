# Stage 12C: spatial/typewell cross-fitとK-best lattice

Stage 12Bは通常well OOFでTop-10 recallを20.1%から82.2%、正解順位中央値を21位から3位へ改善し、expected-offset RMSEも1.693改善した。ただし、通常foldで学習した予測を空間group別に集計しただけでは、地域固有パターンのリークを否定できない。Stage 12Cではemission TCNを6つのspatial blockと5つのtypewell-signature blockで完全に再学習する。

## decoder

各行の61状態log probabilityへ次を加える。

- 隣接状態のjump二乗penalty
- 1 stepで許す最大jump
- Stage 11C surfaceから離れる弱いzero-offset prior
- suffix開始点を既知prefixのanchorへ接続するinitial prior

6個の固定profileだけを評価する。各profileはforward-backward posterior meanを作り、元のrow-wise expected offsetと25〜100%だけ混合する。outer foldのprofileは残りのinner foldsだけで選択され、inner fold RMSE・tail・最大fold劣化の条件を同時に満たさなければpathを適用しない。

通常foldでは、outer-foldが選択したprofileごとに上位16経路も厳密なdynamic programmingで復元する。経路costからposterior weightを計算したK-best meanを、forward-backward posteriorとは別に報告する。Viterbi 1-bestへの即断は最終予測として固定しない。

## promotion gate

- 通常nested pathが0.05以上改善
- spatialとtypewell nested pathが各0.02以上改善
- 3 familyすべてのwell bootstrap上限が0未満
- 通常P90とworst-10% shareが1%超悪化しない
- nested K-bestがrow expected baselineから0.02超悪化しない
- 全family・foldで安全な固定inference profileが存在
- hidden TVTを変更しても特徴とpath入力が不変

合格してもKaggle提出は作らない。次段階で全well emission ensembleを学習し、独自inference packageとしてtest用Notebookへ移植する。
