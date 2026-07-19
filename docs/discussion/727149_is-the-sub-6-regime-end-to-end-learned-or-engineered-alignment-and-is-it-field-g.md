# Is the sub-6 regime end-to-end learned, or engineered alignment? (and is it field-grouped-CV-safe?)

- 投稿者: Murat A. Genc
- 投稿日時: 2026-07-17 23:14:04.853000
- 投票数: 1
- コメント数: 0（取得数: 0）
- トピックID: `727149`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/727149](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/727149)

## 本文

<p>Sharing an honest baseline first: my within-field, group-by-well pipeline (structural datum + particle-filter GR tracking) sits at ~9.9 pooled RMSE on full-773 OOF, squarely in the cluster Georgy and others describe (const 9.04 / line  6.70 oracles). I've independently reproduced that the per-well drift slope is not learnable from legal features (field-grouped OOF R² < 0, shuffle-controlled). So my questions are aimed at the gap between this ~9–10 cluster and the  sub-6 results:</p>
<ol>
<li><p>For the ~5.7-pooled per-well models (group-by-well CV): does the model see the type-well GR-vs-TVT curve as an input channel and learn the match implicitly, or is there an explicit alignment/warp step (DTW with slope limits  particle filter / HMM-Viterbi) that produces the coordinate frame ,  i.e. a candidate TVT(MD) path ,  that the model then refines? In short: is geosteering learned end-to-end, or engineered as preprocessing?</p></li>
<li><p>On the shared bimodal blind spot (several people report independent models converging on the same wrong mode for long cycle-skip stretches, ~±90 ft): does anything break that beyond sequence smoothness? Specifically, do the  train-only formation-top columns (ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA) give an absolute-layer anchor that resolves the ambiguity, or is that error just accepted on the outlier wells?</p></li>
<li><p>On CV realism (the one that decides everything for me): ~5.3–5.8 pooled is reported with random group-by-well, while within-field hold-out for the same method class tops out ~10. Is the difference field-grouping ,  does random  group-by-well leak within-field structure? And do the hidden test wells sit inside the train fields (making within-field interpolation legal) or in held-out fields (making it a leak)?</p></li>
</ol>

## コメント

_コメントなし_
