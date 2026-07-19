# Pointwise GR Makes No Sense

- 投稿者: Rajul_Y
- 投稿日時: 2026-06-30 10:34:26.554000
- 投票数: 2
- コメント数: 1（取得数: 1）
- トピックID: `716289`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/716289](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/716289)

## 本文

<p>I am trying to understand the right framing for using only the horizontal well + its typewell, there is this recurring claim that a strong model can get around ~5 ft pooled row-RMSE using only per-well information, but a ~5 ft result seems to require recovering some of the local wiggle too. For those finding strong per-well-only signal, is the useful object closer to a constrained sequence-alignment / path-likelihood problem than a direct rowwise GR misfit?
When you use the known prefix, do you think of it mostly as:
TVT/O as an anchor,
GR amplitude/offset calibration region,
A local template to fuse with the typewell.
Is denoising horizontal GR materially useful downstream, or mostly cosmetic?
Is the main failure mode of naive GR matching aliasing, or is it the physical path prior being wrong?</p>

## コメント

### コメント 1 — Georgy Mamarin

- 投稿日時: 2026-07-03 06:23:43.853000
- 投票数: -2
- コメントID: `3486899`

<p>Good decomposition — these four are exactly the axes I've been measuring, so here's what I got, with the caveat that it's all train-side.</p>
<p>Sequence alignment vs rowwise misfit: sequence, and it isn't close. The cleanest evidence is an oracle test: let a global GR registration see the true eval GR and grid-search offset and slope of the TVT line — it scores ~31 ft against ~17 for carrying the last value flat, recovering the slope at essentially zero correlation. Pointwise/global GR fit actively hurts, because GR(TVT) repeats. A windowed shape match over a trailing span is what made my particle filter come alive after a single-point emission version sat inert. (A proper bounded-vs-unbounded aligner bake-off is written up for the notebook's next update.)</p>
<p>The known prefix: both an anchor and a calibration region, and the second role is the underrated one. Fitting GR gain/offset on the heel (where TVT is given) and carrying it to the tail takes datum localization from ~8% of wells (flat-surface calibration) to ~80%, essentially the oracle's 82% (a truth-centred scan, so a train-side diagnostic, not a recipe). The prefix is where the amplitude mapping lives.</p>
<p>Denoising: material but modest. A 7-point rolling median (a crude stand-in for the FFT rotation notch from the Problem Breakdown thread — Shrey spotted the rotation, MY0705 proposed the notch) moves that localization from ~80% to ~84% — real, not transformative.</p>
<p>Aliasing vs wrong path prior: aliasing is the disease — on ~12% of wells the decoy a bundle away fits the GR better than the truth even under oracle calibration, the same ±15 ft bimodal datum souldrive derived from the geology — and a tight path prior is the treatment rather than a competing diagnosis, since an unconstrained matcher happily walks into the alias. So "or" is the wrong connective: the prior doesn't fix the tie, it stops you from wandering into it.</p>
<p>Longer version with the measurement cells visible: <a href="https://www.kaggle.com/code/georgymamarin/stop-reforking-the-best-gr-fit-is-the-wrong-depth">where the ROGII error is recoverable vs irreducible</a>.</p>
