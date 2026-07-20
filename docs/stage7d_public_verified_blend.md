# Stage 7D: 検証済み公開OOFのnested blend

Stage 7Cの監査で、fleongg packageには`features.json`と推論用LightGBM 3個しかなく、OOFとfold対応情報がないことを確認した。このbranchはhonest blend検証から除外する。

pilkwang packageには次が揃っている。

- `oof/train_gt.parquet`: ID、well、row、last-known TVT、正解TVT
- family別全3,783,989行OOF
- tree + TCN blend OOF
- postprocessed blend OOF
- 5-fold model、manifest、fold score

Stage 7DではpackageのIDをravaghi baseへ整列し、`target_tvt`とbase `y_true`の最大差が`0.05 ft`以下であることを必須にする。predictionはpackageのdeltaへpackage自身の`last_known_TVT`を加えて絶対TVTへ戻す。

候補branchはpostprocessed blend、raw blend、CatBoost、TCN、HGB、LightGBMである。各branchをravaghi baseへ`1, 2, 5, 10, 15, 20, 30, 40%`混ぜる。各outer foldでは残りのfoldだけでbranchとweightを選び、選択側の改善が`0.02`未満なら補正しない。地理blockでも同じnested選択を独立に行う。

両package prediction自体は通常GroupKFold OOFであり、pilkwangのformation/KNN imputerも空間blockごとに再構築されていない。したがって空間結果はblend weightの地域移転性を調べる追加監査であって、完全なspatial retraining scoreではない。

[130_colab_public_verified_blend_gate.ipynb](../../notebooks/130_colab_public_verified_blend_gate.ipynb)をColabで実行する。これはCPUでよく、学習やKaggle提出は行わない。

`promoted: true`の場合だけ、選ばれたpackage all-train modelとblend weightをSafe MHAへ組み込む。`false`の場合は全OOF最良weightを後付け採用しない。
