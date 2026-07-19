# How to submit your Writeup for the Working Note Award

- 投稿者: María Cruz
- 投稿日時: 2026-06-30 21:24:40.128000
- 投票数: 10
- コメント数: 42（取得数: 42）
- トピックID: `716699`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/716699](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/716699)

## 本文

<p>Hi everyone, 
a couple of weeks ago, we announced the Working Note Awards. If you are planning on submitting your working note, here is how to do it:</p>
<ol>
<li>Navigate to "Your Work", on the left side menu. </li>
<li>Click on the button "Create" that sits right under the page title "Your Work" and click "New Project Writeup". (This is a black button with white letters, different from the white button with a blue + sign and grey letters under the Kaggle logo on the top left corner.)</li>
<li>Create your Writeup. Don't forget to make it public!</li>
<li>Share your writeup on this thread by July 6, at 11:59pm UTC.</li>
</ol>
<p>Please refer to my <a href="https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/709495">previous message on the subject</a> and to the <a href="https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/www.kaggle.com/competitions/rogii-wellbore-geology-prediction/overview/evaluation">Evaluation section</a> in the Overview tab for evaluation criteria and eligibility. </p>
<p>Good luck!</p>
<p>María</p>

## コメント

### コメント 1 — radiant-allomancer

- 投稿日時: 2026-07-13 17:47:31.187000
- 投票数: 1
- コメントID: `3496357`

<p>Sharing my late Working Note with the host's approval above: <strong>When Better CV Scores Worse: A Control-First Geosteering Campaign</strong> — <a href="https://www.kaggle.com/writeups/radiantallomancer/when-better-cv-scores-worse-a-control-first-geost">https://www.kaggle.com/writeups/radiantallomancer/when-better-cv-scores-worse-a-control-first-geost</a></p>
<p>It documents the physical tracker/GRU/trellis solution, eight real CV-to-LB outcomes, controlled negative results, and well-level uncertainty. Thank you for allowing the late submission.</p>

### コメント 2 — Shrey Gandhi

- 投稿日時: 2026-07-06 19:20:10.210000
- 投票数: 6
- コメントID: `3491841`

<p>Sharing my writeup: <a href="https://www.kaggle.com/writeups/shreygandhi/stride-a-joint-posterior-over-depth-and-distance">https://www.kaggle.com/writeups/shreygandhi/stride-a-joint-posterior-over-depth-and-distance</a></p>

#### コメント 2.1 — Andrey Chankin

- 投稿日時: 2026-07-18 21:15:40.090000
- 投票数: 0
- コメントID: `3500216`

<blockquote>
  <p>Sharing my writeup: <a href="https://www.kaggle.com/writeups/shreygandhi/stride-a-joint-posterior-over-depth-and-distance">https://www.kaggle.com/writeups/shreygandhi/stride-a-joint-posterior-over-depth-and-distance</a></p>
</blockquote>
<p>Hi <a href="https://www.kaggle.com/shreygandhi">@shreygandhi</a>, shame your writeup didnt win, considering highest LB position.</p>
<p>Do you mind explaining this little further </p>
<blockquote>
  <p>A typewell's GR readings and this well's own GR readings can sit at slightly different levels even for identical rock, because each logging run has its own instrument offset. Before decoding, I nudge the typewell's GR-versus-depth curve toward this well's own observed GR, using only the known section where both are available at the same true depths. Without that adjustment, the likelihood above would be comparing this well's real log against a reference that might be running systematically high or low, which can drag a candidate's whole depth register off even when its shape matches well.</p>
</blockquote>

#### コメント 2.2 — Shrey Gandhi

- 投稿日時: 2026-07-19 06:24:24.473000
- 投票数: 0
- コメントID: `3500342`

<p>Hey <a href="https://www.kaggle.com/bluepill">@bluepill</a> , It is what it is. 
Its like using the know sections GR values to blend the right information in for the whole typewell. Doing this will reduce drift and improve alignment. 
All the best to you!</p>

### コメント 3 — Malyshev Danil

- 投稿日時: 2026-07-06 23:45:50.847000
- 投票数: 4
- コメントID: `3492271`

<p>Sharing our Working Note: "The Wiggle Is Free, the Trend Is the Wall" — <a href="https://kaggle.com/writeups/malyshevdanil/the-wiggle-is-free-the-trend-is-the-wall">https://kaggle.com/writeups/malyshevdanil/the-wiggle-is-free-the-trend-is-the-wall</a>
An exact decomposition (TVT = surface − Z) shows the high-frequency part of the target is free — it's exactly the known trajectory depth, no model needed — collapsing the whole task to recovering one smooth per-well surface trend. Twelve independent measurements (including a coherence-spectrum measurement of the acquisition instrument itself) show that trend is broadband-irreducible: the eval-zone dip is effectively a new, unobserved constant of integration, not encoded anywhere in the GR log.</p>
<p>We ran more than 60 experiments testing a wide range of different models and approaches — GR-matching schemes, particle-filter variants, GBM/classical-ML blends, denoising, human-markup reverse-engineering, and a dedicated sweep of 22 distinct neural network architectures (MDN, transformers, 2D misfit-SDF, contrastive matchers, learned Bayesian filters, and more) — and mapped the specific failure mechanism of each. Every from-scratch neural model floors at the isolated-PF level (~11 ft) regardless of design, which is itself a structural, not incidental, finding.</p>
<p>Our own real-LB progression: 7.080 → 6.836 (stacking a decorrelated neural corrector with a robust physics post-process) → 6.794 (transferring a cross-validated hyperparameter back onto the submission) — every step confirmed by direct resubmission, 5-fold CV, and a spatial-block (LSO) holdout, not just proxy validation. We also found a hard limit: a third, individually cross-fit-validated correction breaks a combination that had cleanly stacked two — corrections don't compose indefinitely even when each is honestly validated on its own. Full negative-result catalogue is in the note. Feedback welcome.</p>

### コメント 4 — daulettoibazar

- 投稿日時: 2026-07-05 21:49:58.327000
- 投票数: 3
- コメントID: `3489825`

<p>Here's my write-up, a summary of my approach and results:
<a href="https://www.kaggle.com/writeups/daulettoibazar/working-note-our-solution-the-failures-behind-it">https://www.kaggle.com/writeups/daulettoibazar/working-note-our-solution-the-failures-behind-it</a></p>

#### コメント 4.1 — Georgy Mamarin

- 投稿日時: 2026-07-06 03:56:58.570000
- 投票数: -5
- コメントID: `3490027`

<p>The LSO axis is the load-bearing idea here: a well-grouped CV can't see neighbour over-fitting because a held-out well keeps its trained neighbours, so leaving whole spatial blocks out is the only split that tracks the board. Your month of seven CV-better submissions that didn't move public until LSO moved with them is the most concrete CV→LB-mirage story I've read. Two more that match my side — your selector handed the true label ranking the correct candidate only ~20% of the time is the "representable but not selectable" wall exactly, and your leak-check (hand each well a different well's alignment evidence and see if the gain survives) is the shuffle-null control every post-processor on this task needs. The two physical tracker fixes being your only real board moves, after a month of neighbour features, is the whole lesson in one arc.</p>

### コメント 5 — Vishal Kishore

- 投稿日時: 2026-07-06 20:38:31.143000
- 投票数: 2
- コメントID: `3492014`

<p>Sharing my writeup (SEGN: Autoregressive Cost-Map View of TVT) - <a href="https://kaggle.com/writeups/kishorevishal/segn-an-autoregressive-cost-map-view-of-tvt">https://kaggle.com/writeups/kishorevishal/segn-an-autoregressive-cost-map-view-of-tvt</a></p>

### コメント 6 — Anthony Yanza

- 投稿日時: 2026-07-02 22:49:16.563000
- 投票数: 1
- コメントID: `3486744`

<p>Sharing my working note here: "The Acquisition Physics of the ROGII Tail: A Measured Tool Response Operator" — <a href="https://www.kaggle.com/writeups/anthonyyanza/the-acquisition-physics-of-the-rogii-tail-a-measu">https://www.kaggle.com/writeups/anthonyyanza/the-acquisition-physics-of-the-rogii-tail-a-measu</a></p>
<p>It measures the wireline-to-LWD gamma ray response operator directly from the data, including a section-resolved coherence spectrum showing the fine (5-32 ft) Milankovitch band is nearly absent from the toe rows the metric scores, which bounds the community's bimodal-datum picture with acquisition physics. Also includes an exhaustive study of 30+ approaches, three new negative results, and an explicit correction of a mistaken distance assumption that had closed off the structural/neighbor family. Happy to discuss.</p>

#### コメント 6.1 — Georgy Mamarin

- 投稿日時: 2026-07-06 03:49:38.603000
- 投票数: -2
- コメントID: `3490016`

<p>The coherence spectrum is the measurement I hadn't seen anyone make, and it's the right one — 0.03–0.11 across the 5–32 ft band on the toe turns "the likelihood is degenerate" from an observation into a mechanism: the discriminating band sits below the instrument's floor exactly where the metric scores. Two things line up from my side — your joint dip+datum decode at 81 ft with the correct branch only 31% of the time is the same wall I hit from the correlation angle (the margin between the two modes carries almost no information about which is right), and "the correction exists but the information to select it doesn't travel with the well" is the field-grouped OOF R²<0 result stated better than I stated it. Also appreciated the clean withdrawal of the ~16 kft distance assumption once the score-probe put real spacing near 1 kft.</p>

##### コメント 6.1.1 — Anthony Yanza

- 投稿日時: 2026-07-07 04:10:57.927000
- 投票数: 0
- コメントID: `3492559`

<p>Thanks, Georgy. I really appreciated this read. Your point about the joint dip/datum decode and the branch margin is exactly the connection I was trying to make: the structure is representable, but the legal signal often does not carry the selection key. </p>
<p>And agreed on the distance correction. That was the important audit fix for me; the score-probe moved the neighbor story from a confident explanation back into an open question. I’m glad the coherence measurement made the mechanism sharper rather than just adding another failed model to the pile.</p>

### コメント 7 — joethanyoung

- 投稿日時: 2026-07-06 16:31:48.957000
- 投票数: -1
- コメントID: `3491282`

<p>Sharing my Working Note: "Pricing the Missing Sign: Forty Audits on the ROGII Tail" — <a href="https://www.kaggle.com/writeups/zeoyoungy/pricing-the-missing-sign-forty-audits-on-the-rogi">https://www.kaggle.com/writeups/zeoyoungy/pricing-the-missing-sign-forty-audits-on-the-rogi</a></p>
<p>A measurement report on what the legal inputs can and cannot recover: error anatomy (the residual is a per-well datum problem), the two-knobs argument for why hedging is the squared-loss optimum under a missing sign, a sign-observable sweep with deployment gates, and quantified leaderboard-noise bands. Negative results included with their controls. Feedback welcome.</p>

### コメント 8 — lllllllll

- 投稿日時: 2026-07-06 15:14:03.497000
- 投票数: -1
- コメントID: `3491032`

<p>I frame TVT prediction as an autoregressive sequence problem: a GRU predicts the per-step change in TVT, and the well is rolled out step by step from the last known anchor.  <a href="https://www.kaggle.com/writeups/crabxmz/autoregressive-gru-with-typewell-correlation-featu">https://www.kaggle.com/writeups/crabxmz/autoregressive-gru-with-typewell-correlation-featu</a></p>

### コメント 9 — Sergey Subbotin

- 投稿日時: 2026-07-06 15:05:32.977000
- 投票数: -1
- コメントID: `3491013`

<p>Sharing my working note: "Measuring What Is Learnable: A Validation-First ROGII Campaign"
<a href="https://www.kaggle.com/writeups/ssubbotin/measuring-what-is-learnable-a-validation-first-ro">https://www.kaggle.com/writeups/ssubbotin/measuring-what-is-learnable-a-validation-first-ro</a></p>
<p>It documents a validation-first campaign: a CV-to-LB transfer model plus a public-LB (52-well draw) variance simulator, a non-negative QP (BLUE) blend of decorrelated trajectory decoders, and a 12-entry map of negative results locating where the total-GR signal ends, with a mechanism-level account of the ~8.3 ft leak-free ceiling. Feedback welcome.</p>

### コメント 10 — Ramesh Arvind

- 投稿日時: 2026-07-06 14:53:27.883000
- 投票数: -1
- コメントID: `3490980`

<p>Sharing my writeup, a summary of results here: <a href="https://www.kaggle.com/writeups/rameshln/bayesian-geosteering-for-rogii-particle-filters-a">https://www.kaggle.com/writeups/rameshln/bayesian-geosteering-for-rogii-particle-filters-a</a></p>

### コメント 11 — Georgy Mamarin

- 投稿日時: 2026-07-06 10:29:32.663000
- 投票数: -3
- コメントID: `3490520`

<p>Sharing my Working Note, Fork the ruler, not the model:
<a href="https://www.kaggle.com/writeups/georgymamarin/fork-the-ruler-not-the-model">https://www.kaggle.com/writeups/georgymamarin/fork-the-ruler-not-the-model</a></p>
<p>It maps where the ROGII surface is recoverable from the given data and where it is not: a per-well offset plus a piecewise dip carries the median wells to ~3-5 ft, and an irreducible bimodal datum sits on the ~10% of wells that hold ~40% of the squared error. The result I'd flag: the calibration degeneracy is not the wall. A legal heel-calibration recovers ~80% localization (near the oracle's 82%), so the open legal problem moves from the datum to the per-well slope. It also ships a small harness (oracle-ceiling ladder, tail concentration, leave-group-out wall test) you can point at your own OOF. Thanks to everyone whose notes and EDA it builds on. Feedback very welcome.</p>

### コメント 12 — Bernardus

- 投稿日時: 2026-07-06 09:57:06.723000
- 投票数: -1
- コメントID: `3490476`

<p>Here is my writeup : <a href="https://www.kaggle.com/writeups/bernubritz/the-predictability-boundary-a-study-in-spatial-le">https://www.kaggle.com/writeups/bernubritz/the-predictability-boundary-a-study-in-spatial-le</a></p>

### コメント 13 — Julian Camilo Villa

- 投稿日時: 2026-07-04 04:14:58.953000
- 投票数: -2
- コメントID: `3487596`

<p><a href="https://kaggle.com/writeups/juliancamilovilla/what-did-and-did-not-work-honest">https://kaggle.com/writeups/juliancamilovilla/what-did-and-did-not-work-honest</a>
An honest particle-filter pipeline that predicts wellbore TVT by tracking the structural surface S = TVT + Z from gamma-ray, paired with a map of what does and does not generalize on this task — including why the best GR fit points at the wrong depth (the bimodal datum) and why most apparent gains are just particle-filter seed noise</p>

#### コメント 13.1 — Georgy Mamarin

- 投稿日時: 2026-07-06 03:55:23.417000
- 投票数: -5
- コメントID: `3490025`

<p>Thanks for building on the diagnostic and for the careful credit — but the part I want to point at is your own result, because it sharpens something I only asserted. You tried the bimodal heel-calibrated hedge and it hurt your pipeline, where it helped a couple of the GBM-based notes, and you attribute it to your PF's particle spread already carrying the uncertainty the hedge would add — which is why on your base it's redundant and on a confounded shift-scan harmful. That's the cleanest demonstration I've seen that the hedge is base-dependent, not a universal lever — a variance move that only pays where the base isn't already spending that variance. The physics post-process being your largest honest gain (−0.57) is also a nice counterweight to everyone chasing the matcher.</p>

##### コメント 13.1.1 — Julian Camilo Villa

- 投稿日時: 2026-07-06 16:11:54.440000
- 投票数: 0
- コメントID: `3491230`

<p>Thank you — "a variance move that only pays where the base isn't already spending that variance" is sharper than my phrasing, and I'll be borrowing it. Two numbers that make the base-dependence even cleaner: I also re-tested the gentle gated form the GBM pipelines report gains from (two near-tied scan minima + cross-tracker disagreement, α=0.2 pull to the midpoint). On my base it gains a real −0.15 on a random hold-out that inverts on both distribution-shift splits (+0.02 spatial, +0.16 high-drift) — and the disagreement gate is load-bearing: without it the same hedge is worse on every split. So on my base the scan survives only as a trigger, never as a target. One refinement to the redundancy reading: it's not just the PF spread — the post-process and meta-layer already spend the smooth-drift risk budget the hedge would spend again. Same budget, two spenders.</p>

### コメント 14 — radiant-allomancer

- 投稿日時: 2026-07-13 12:37:42.530000
- 投票数: 0
- コメントID: `3496219`

<p>Hi <a href="https://www.kaggle.com/macruzbar">@macruzbar</a>, our team was in the public Medal Zone at the July 6 deadline, but we missed the separate Working Note submission step. The official timeline says notes had to be shared by July 6 at 11:59pm UTC. Could you please confirm whether late Working Note submissions are categorically ineligible, or whether the host would still consider one? We understand if the deadline is final. Thank you.</p>

#### コメント 14.1 — María Cruz

- 投稿日時: 2026-07-13 17:16:45.533000
- 投票数: 0
- コメントID: `3496341`

<p>HI <a href="https://www.kaggle.com/radiantallomancer">@radiantallomancer</a> -- I checked with the host and they confirmed it's ok to do a late submission. Good luck!</p>

##### コメント 14.1.1 — FOYSAL

- 投稿日時: 2026-07-13 18:00:29.903000
- 投票数: 0
- コメントID: `3496363`

<p>Hello <a href="https://www.kaggle.com/macruzbar">@macruzbar</a> </p>
<p>Will we find out which two write-ups were selected as the best only after the competition ends, or will they be announced during the competition?</p>

##### コメント 14.1.2 — María Cruz

- 投稿日時: 2026-07-13 19:09:18.183000
- 投票数: 0
- コメントID: `3496388`

<p>Hi <a href="https://www.kaggle.com/foysalemonshanto">@foysalemonshanto</a> -- winners will be announced during the competition. Good luck!</p>

##### コメント 14.1.3 — Andrey Chankin

- 投稿日時: 2026-07-18 20:43:18.790000
- 投票数: 0
- コメントID: `3500206`

<blockquote>
  <p>HI <a href="https://www.kaggle.com/radiantallomancer">@radiantallomancer</a> -- I checked with the host and they confirmed it's ok to do a late submission. Good luck!</p>
</blockquote>
<p>Thats quite unfair for other teams</p>

### コメント 15 — o_O

- 投稿日時: 2026-07-07 04:36:49.883000
- 投票数: 0
- コメントID: `3492611`

<p>Drill baby drill: <a href="https://kaggle.com/writeups/kegelkaggle/guilty-until-proven-validated">https://kaggle.com/writeups/kegelkaggle/guilty-until-proven-validated</a>.</p>

### コメント 16 — Rohan Vinaik

- 投稿日時: 2026-07-06 21:24:24.823000
- 投票数: 0
- コメントID: `3492092`

<p>Sharing my writeup!
<a href="https://www.kaggle.com/writeups/rohanvinaik/law-as-architecture#3492081">https://www.kaggle.com/writeups/rohanvinaik/law-as-architecture#3492081</a></p>

### コメント 17 — 藤田　佑

- 投稿日時: 2026-07-06 19:29:56.947000
- 投票数: 0
- コメントID: `3491861`

<p>Here is my Working Note submission:</p>
<p>Where the Information Ends<br>
<a href="https://www.kaggle.com/writeups/yuu111111111/a-falsification-program-on-gr-based-geosteering">https://www.kaggle.com/writeups/yuu111111111/a-falsification-program-on-gr-based-geosteering</a></p>
<p>Public LB: 7.106</p>
<p>A falsification-focused note on where my PF/stacker/spatial pipeline stopped improving. It reports 51 controlled experiments, decomposes the remaining error into LINE vs SHAPE, and focuses on the detection-vs-actuation gap: many failure cases can be detected, but the legal signals were not reliable enough to choose the corrective direction.</p>

### コメント 18 — Cefiyana

- 投稿日時: 2026-07-06 18:35:04.483000
- 投票数: 0
- コメントID: `3491724`

<p>Hi everyone, thanks to the organizers for hosting such a challenging competition. </p>
<p>I've just published a detailed write-up of my solution, the "Clover" architecture (20.903 RMSE). The manuscript details the transition to a Physics-Informed Machine Learning (PIML) framework, specifically focusing on how constrained Dynamic Time Warping (DTW) and spatial calculus can bound thermodynamic signal noise within physical laws. </p>
<p>You can read the full breakdown here: <a href="https://kaggle.com/writeups/cefiyana/the-clover-architecture-deterministic-piml-in-wel">https://kaggle.com/writeups/cefiyana/the-clover-architecture-deterministic-piml-in-wel</a></p>
<p>I'd love to hear your thoughts, especially regarding the "Integral Drift" limitation I mentioned in the conclusion. Happy reading!</p>

### コメント 19 — Handa WANG

- 投稿日時: 2026-07-06 12:27:22.700000
- 投票数: 0
- コメントID: `3490702`

<p>This is my first attempt at writing a write-up. It is based on a U-Net solution, but I did not achieve a particularly strong result. I would be very grateful for any advice or feedback from the community.
<a href="https://www.kaggle.com/writeups/handawang/compact-1d-u-net-distillation-from-top-k-wellbore">https://www.kaggle.com/writeups/handawang/compact-1d-u-net-distillation-from-top-k-wellbore</a></p>

### コメント 20 — Myckel Uribe

- 投稿日時: 2026-07-05 05:05:40.093000
- 投票数: 0
- コメントID: `3488591`

<p>Hello, I have published my Working Note for the ROGII Wellbore Geology Prediction competition:</p>
<p><strong>The Predictability Boundary of Pre-Drill Stratigraphic Targeting</strong>
<a href="https://www.kaggle.com/writeups/myckeluribe/the-predictability-boundary-of-pre-drill-stratigra">https://www.kaggle.com/writeups/myckeluribe/the-predictability-boundary-of-pre-drill-stratigra</a></p>
<p>The note covers a physical particle-filter stack, leakage controls, identifiability and selection-frontier analysis, twenty-eight gated experiments, leaderboard-family attribution, and an operational geosteering uncertainty translation.</p>
<p>Thanks to the organizers and the community for the competition and discussions.</p>

#### コメント 20.1 — Georgy Mamarin

- 投稿日時: 2026-07-06 03:51:42.683000
- 投票数: -3
- コメントID: `3490021`

<p>The identifiability lemma is the cleanest statement I've seen of why the datum is structurally unrecoverable rather than merely unrecovered — the offset as the constant of integration, a 0.1° dip error integrating to ~8.7 ft over 5 kft. And the metric-variance decomposition (Var(MSE) ∝ σ⁴_offset · Σn²/N², long wells dominating) is the rigorous version of the seed-band argument the rest of us wave our hands at: it says the several-tenths CV↔LB swings are expected sampling variance, not skill, without accusing anyone of leaking. Your selection frontier lands where several of us did — a learned posterior-mean scorer reaching top-1 only ~21% and then degrading a strong baseline +0.80 live is the hedge being a variance move that stops helping once the base is finer than the corrector's resolution. Thanks for the citation.</p>

### コメント 21 — LeeDongHyuk

- 投稿日時: 2026-07-03 14:06:56.470000
- 投票数: 0
- コメントID: `3487201`

<p>Sharing our working note: "Anatomy of an Unobservable Datum: A Systematic Investigation of the ROGII Wellbore Prediction Problem" — <a href="https://www.kaggle.com/writeups/odyssey189/anatomy-of-an-unobservable-datum">https://www.kaggle.com/writeups/odyssey189/anatomy-of-an-unobservable-datum</a></p>
<p>The note is organized around a measured negative result: after a strong GBM + particle-filter base, ~60% of the remaining squared error is a per-well datum offset that we show is unobservable from the legal inputs (seven independent gates, oracle ceilings, and corroborating theory from navigation observability and geosteering literature). The constructive half covers what works once you accept that: selective (overlap-gated) IRLS, a bimodal-datum midpoint hedge (our largest verified gain, −0.16 public), L1-objective diversity, duplicate forensics, and the detection-vs-actuation dissociation in uncertainty estimation. Feedback very welcome!</p>

#### コメント 21.1 — Georgy Mamarin

- 投稿日時: 2026-07-06 03:53:52.167000
- 投票数: -2
- コメントID: `3490023`

<p>The James–Stein shrink fraction coming out at exactly 0.000 is my favourite result in this — an estimator that refuses to shrink is telling you the SNR is zero, a cleaner proof of unobservability than any single gate because the estimator diagnoses itself. And grounding it in navigation observability (constant bias states unobservable from increments) and the geosteering patents that output a posterior over the offset rather than a point is the right literature — it reframes "we couldn't predict it" as "the field's best practice is to not answer this question." Your detection-works / actuation-fails dissociation (AUC 0.69 vs direction R² −0.16) is the same split I read as a trust-gate: the spread tells you where you're wrong, never which way. Independent convergence down to the 0.0065 ft identity is reassuring.</p>

### コメント 22 — FOYSAL

- 投稿日時: 2026-07-02 15:29:15.433000
- 投票数: 0
- コメントID: `3486481`

<p>Hello,
I have published my Working Note for the ROGII Wellbore Geology Prediction competition:</p>
<p>ROGII Wellbore Prediction: Anchors, GR Alignment, and Guarded Geosteering <a href="https://www.kaggle.com/writeups/foysalemonshanto/rogii-wellbore-prediction-anchors-gr-alignment">https://www.kaggle.com/writeups/foysalemonshanto/rogii-wellbore-prediction-anchors-gr-alignment</a></p>
<p>The note focuses on the validation work behind the solution: why last-known TVT is such a strong anchor, where GR/typewell alignment helps, why naive slope extrapolation and ungated GR matching often fail, and how guarded contact/geosteering ideas shaped the final approach.</p>
<p>Feedback is very welcome. Thanks to the organizers and the community for the competition and discussions.</p>

#### コメント 22.1 — Bernardus

- 投稿日時: 2026-07-02 18:00:41.430000
- 投票数: 0
- コメントID: `3486610`

<p>Deadline is July 6th. </p>

#### コメント 22.2 — Julian Camilo Villa

- 投稿日時: 2026-07-06 03:30:49.807000
- 投票数: 0
- コメントID: `3490010`

<p>Hello
Link is broken or is not public</p>

#### コメント 22.3 — María Cruz

- 投稿日時: 2026-07-06 16:06:59.660000
- 投票数: 0
- コメントID: `3491210`

<p>Hi <a href="https://www.kaggle.com/foysalemonshanto">@foysalemonshanto</a> -- can you please ensure the link is publicly accessible and typed correctly? I was not able to access your Writeup.</p>

##### コメント 22.3.1 — Rokaiya Somapti

- 投稿日時: 2026-07-07 00:03:08
- 投票数: 0
- コメントID: `3492286`

<p>We had kept it private. Now it's public.</p>

### コメント 23 — Unknown

- 投稿日時: 2026-07-06 23:42:54.403000
- 投票数: 0
- コメントID: `3492270`

_本文なし_

### コメント 24 — Unknown

- 投稿日時: 2026-07-06 18:06:46.497000
- 投票数: 1
- コメントID: `3491629`

_本文なし_

#### コメント 24.1 — daulettoibazar

- 投稿日時: 2026-07-06 18:58:43.190000
- 投票数: 0
- コメントID: `3491796`

<p>I wanted to flag that this writeup suspiciously matching mine (<a href="https://www.kaggle.com/writeups/daulettoibazar/working-note-our-solution-the-failures-behind-it">https://www.kaggle.com/writeups/daulettoibazar/working-note-our-solution-the-failures-behind-it</a>, posted 21 hours ago) far too closely to be coincidental: identical section titles, the same structure, and many word-for-word sentences. <a href="https://www.kaggle.com/macruzbar">@macruzbar</a> could you take a look when you have a chance? Happy to point out the specific overlapping sections.</p>
