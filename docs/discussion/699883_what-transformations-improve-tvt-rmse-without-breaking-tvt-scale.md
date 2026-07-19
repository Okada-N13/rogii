# What transformations improve TVT RMSE without breaking TVT scale?

- 投稿者: POR160893
- 投稿日時: 2026-05-15 16:10:57.955000
- 投票数: 8
- コメント数: 2（取得数: 2）
- トピックID: `699883`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699883](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699883)

## 本文

<p>After ~35–40 phases of analogue/retrieval/residual experiments, I think the main unresolved question is no longer “how do we fit GR better?”, but:</p>
<p>What transformations of a strong TVT anchor actually improve hidden TVT RMSE without destroying absolute TVT scale?</p>
<p>A lot of my earlier experiments implicitly assumed:</p>
<ul>
<li>better GR match -> better TVT</li>
<li>better local sequence analogue -> better submission</li>
</ul>
<p>But discussion here (especially around inverse problems / priors) suggests this is not necessarily true.</p>
<p>What I’m seeing empirically:</p>
<ul>
<li>Strong anchor submissions (Phase23D-style smooth constrained solutions) remain surprisingly robust.</li>
<li>Aggressive free-form transformations can catastrophically fail even when pseudo-validation looks good.</li>
<li>Small flatten/compression variants sometimes move LB slightly, but signal is weak.</li>
<li>Preserving global TVT statistics (mean/std/range/drift) seems extremely important.</li>
<li>Local GR resemblance alone does not appear sufficient.</li>
</ul>
<p>So I’m trying to understand the correct search space.</p>
<p>For those experimenting successfully:</p>
<ol>
<li>Are you mostly applying:</li>
</ol>
<ul>
<li>affine transforms?</li>
<li>local warping?</li>
<li>slope regularization?</li>
<li>residual learning?</li>
<li>monotonicity priors?</li>
<li>MD-domain smoothing?</li>
<li>sequence transplantation?</li>
</ul>
<ol>
<li>How tightly are you constraining:</li>
</ol>
<ul>
<li>per-well mean?</li>
<li>std/range?</li>
<li>drift?</li>
<li>curvature/roughness?</li>
</ul>
<ol>
<li>Has anyone found evidence that:</li>
</ol>
<ul>
<li>hidden RMSE rewards smoother geological priors more than local GR fit?</li>
<li>preserving absolute TVT scale matters more than matching local structure?</li>
</ul>
<ol>
<li>Most importantly:
If you start from a “good anchor”, what transformations have actually improved LB consistently rather than randomly?</li>
</ol>
<p>At this point I suspect the competition is largely about learning the correct prior space for TVT, not maximizing GR similarity.</p>

## コメント

### コメント 1 — hengck23

- 投稿日時: 2026-05-15 21:13:47.257000
- 投票数: 3
- コメントID: `3458439`

<p>code and lesson (lecture notes) <a href="https://github.com/geosteering-no/inversion_school_geosteering/tree/main">https://github.com/geosteering-no/inversion_school_geosteering/tree/main</a></p>
<p>the above answers your questions.</p>
<p>1) you need to find a solution without GR first (i suggest linear prior, some public code use knn on ancc plane, the lecture suggests piecewise linear or "anything that is geologically correct", the competition host mentions about nearby wells having the same dip)</p>
<p>2) then fit GR such that the resulting TVT is close to the prior (see lecture slides). Here starsteer patent suggests choosing NCC stretching and windows from some pre-defined "dips" and " large segment sizes", which is a form of constraint too. </p>
<p>in short, it is about lowest GR rmse  under what constraints?</p>
<p>the CNN method in the lecture note did not explicitly mnimize GR loss nor do correlation NCC, becuase the loss they minimize is TVT mse </p>
<p>The public solution did it slightly differently. instead of using these local results of GR NCC directly (or NCC-based PF/k-beams), these are fed into a model that uses them as features to predict TVT. </p>

### コメント 2 — hengck23

- 投稿日時: 2026-05-15 19:33:32.320000
- 投票数: 2
- コメントID: `3458407`

<p>You can still use GR for fitting, but the goal should be to fit the geological structure represented by the GR log, not merely the raw GR values.  </p>
<p>For example, the peaks and valleys in a GR log may carry the most important geological information. They might make up only around 10% of the signal, but matching that 10% correctly can matter more than matching the other 90%.</p>
