# A geophysicist's take: domain priors + Q-3D tortuosity (public notebook + repo)

- 投稿者: Matteo Niccoli
- 投稿日時: 2026-05-21 13:41:20.677000
- 投票数: 20
- コメント数: 0（取得数: 0）
- トピックID: `702131`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/702131](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/702131)

## 本文

<h1>Kaggle discussion post - ROGII competition</h1>
<hr>
<p>Sharing a notebook and a public repo that approach this competition from a geological reasoning angle rather than pure ML optimization.</p>
<p><strong>Notebook:</strong> <a href="https://www.kaggle.com/code/mycarta/rogii-wellbore-geology-prediction-toolkit">https://www.kaggle.com/code/mycarta/rogii-wellbore-geology-prediction-toolkit</a></p>
<p><strong>GitHub repo:</strong>  <a href="https://github.com/mycarta/rogii-geosteering-toolkit">https://github.com/mycarta/rogii-geosteering-toolkit</a></p>
<p>Single LightGBM, no particle filters or stacking. Mid-pack on the leaderboard, but the findings may be useful to others working on this problem:</p>
<ul>
<li><p><strong>Within-well TVT-Z decoupling.</strong> The global TVT-vs-Z correlation is r = -0.96, but within a single lateral it's essentially zero (mean slope +0.057). The global signal is cross-well structural elevation dominated by build-section geometry. Features based on the global relationship don't work until you account for this.</p></li>
<li><p><strong>Q-3D tortuosity (Jing et al. 2022) was the most useful domain feature</strong> in the ablation (-0.107 RMSE). High tortuosity = active steering = formation deviating from plan.</p></li>
<li><p><strong>Signed drilling azimuth matters.</strong> Opposite directions along the same line see the formation in opposite sequence. The updip/downdip distinction is real and the model uses it via sin/cos azimuth encoding paired with dZ/dMD.</p></li>
<li><p><strong>Well-level AEON features (Catch22 + ClaSP) made the model worse</strong> (+0.476 RMSE). Well-level features under GroupKFold overfit on cross-well noise. Documented as a negative result in the ablation table.</p></li>
<li><p><strong>Considered Verde's BlockKFold for spatial CV, rejected it.</strong> Validation wells are spatially interleaved with training (interpolation, not extrapolation), so spatial blocking is more pessimistic than the actual test condition. Used StratifiedGroupKFold stratified by signed azimuth, median TVT, and spatial location instead.</p></li>
</ul>
<p>The notebook includes a cumulative ablation table, Phase 1 spatial recon figures (well center map, azimuth rose, lateral slope distribution), a MASS-equivalent-to-NCC pedagogical demo, and a Q-3D tortuosity visualization.</p>
<p>The GitHub repo has the reusable toolkit modules (wellbore tortuosity from XYZ trajectories, log despiking, sliding distance correlation) and two methodology write-ups on the TVT-Z decoupling finding and the AEON evaluation.</p>
<p>Feedback welcome, especially from anyone with geosteering or horizontal drilling experience. What domain features would you add?</p>

## コメント

_コメントなし_
