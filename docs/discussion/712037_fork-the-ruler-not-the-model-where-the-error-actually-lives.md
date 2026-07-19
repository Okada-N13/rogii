# Fork the ruler, not the model — where the error actually lives

- 投稿者: Georgy Mamarin
- 投稿日時: 2026-06-22 10:03:21.921000
- 投票数: 23
- コメント数: 29（取得数: 29）
- トピックID: `712037`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/712037](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/712037)

## 本文

<p>Most public notebooks here fork one strong pipeline and cluster around ~7.2. Instead of adding another fork, I measured how much of the remaining error is even recoverable from the data we are given — all on held-out wells.</p>
<ul>
<li>The per-row target reduces to one number per well, a drift slope (the surface is ~99% linear).</li>
<li>That slope is not learnable from the legal features I tried on unseen fields (field-grouped OOF R² below zero, shuffle-controlled).</li>
<li>Why: GR carries real coarse signal (a particle filter gets ~16 ft down to ~10), but it can't pin the fine per-well slope through a legal calibration — a measured degeneracy, not a property to chase.</li>
<li>Plus two traps that look like progress: the CV→LB mirage and seed/refork variance.</li>
</ul>
<p>It comes with a fork-and-go harness (oracle_ceiling + wall_test) that isn't wellbore-specific: drop your own out-of-fold predictions in, read your recoverable ceiling, and run the wall test — leave-one-group-out with a shuffle-null control — on your features.</p>
<p>Notebook: <a href="https://www.kaggle.com/code/georgymamarin/fork-the-ruler-not-the-model">Fork the ruler, not the model</a></p>
<p>If you have found a field-grouped, leak-free signal that predicts the per-well drift, I would like to see it. I will run the harness on it.</p>
<p><strong>Corrections & updates.</strong> Two early corrections from the v2 update, both still standing: GR isn't legally <em>unusable</em> — a particle filter turns it into real signal (~16→10 ft), it just can't pin the fine per-well slope; and the public board isn't leaky — the visible example wells are replaced by ~200 hidden test wells at scoring (thanks Ioannis M). Since then: a three-aligner bake-off (only the particle filter survives, as a trust-gate feature), a ±60 ft bounded-search guardrail, wharekawa's survey-density mechanism for the §6 cross-well failure (credited), and — latest — §9's harness made competition-agnostic and the notebook retitled <em>Fork the ruler, not the model</em>.</p>

## コメント

### コメント 1 — radiant-allomancer

- 投稿日時: 2026-07-13 14:07:32.730000
- 投票数: -6
- コメントID: `3496253`

<p><a href="https://www.kaggle.com/Tucker">@Tucker</a> Arrants Your public comments say the ~5.4 single model is non-tabular and uses only each well plus its typewell. To avoid asking for proprietary details: is the representation closest to (a) a 2-D GR/typewell alignment image, (b) a 1-D sequence model over the lateral, or (c) training-free per-well optimization/state-space? Even just the broad family would help distinguish a reproduction bug from a dead direction. Thanks.</p>

### コメント 2 — Georgy Mamarin

- 投稿日時: 2026-07-04 09:23:06.783000
- 投票数: 0
- コメントID: `3487796`

<p>The §9 harness stopped being wellbore-shaped. <code>wall_test</code> now takes any group column, so the leave-one-group-out + shuffle-null check runs on CV folds, entity ids, or time blocks — not just heel coordinates. Point it at your own OOF: a real feature clears zero where a leaking one doesn't, and you read your own oracle-ceiling floor and worst-decile tail alongside it. If a feature you trust dies under it, better there than on the private split. (The notebook is retitled <em>Fork the ruler, not the model</em> now; §9 is the headline.)</p>

### コメント 3 — Tucker Arrants

- 投稿日時: 2026-06-22 13:53:31.527000
- 投票数: 12
- コメントID: `3478241`

<p>Public notebooks CV are quite high and they are just overfitting the leaderboard with seed noise. You can get a CV ~5ft with a single model and maybe even under 4ft. No idea what you can do with ensembling, haven't tried yet.</p>

#### コメント 3.1 — Rishikesh Jani

- 投稿日時: 2026-06-22 14:33:57.407000
- 投票数: 4
- コメントID: `3478277`

<p>Sounds just like what happened in the BirdCLEF+ competition towards the end. </p>

##### コメント 3.1.1 — Georgy Mamarin

- 投稿日時: 2026-06-22 17:38:31.697000
- 投票数: -4
- コメントID: `3478436`

<p>Good analogy — BirdCLEF+ had the same end-game, where the blends just reshuffle on a noisy private split. That's the seed-variance half of trap #2.</p>

##### コメント 3.1.2 — Shrey Gandhi

- 投稿日時: 2026-06-23 10:03:36.350000
- 投票数: 0
- コメントID: `3479052`

<p>Hey <a href="https://www.kaggle.com/rishikeshjani">@rishikeshjani</a> & <a href="https://www.kaggle.com/georgymamarin">@georgymamarin</a> , What happened in BirdCLEF+, how can we prevent it? 😁</p>

##### コメント 3.1.3 — Georgy Mamarin

- 投稿日時: 2026-06-23 12:18:48.633000
- 投票数: -4
- コメントID: `3479134`

<p>Good question. The short of BirdCLEF+ near the end: the private split was small and noisy, so late blends mostly reshuffled rank by seed luck — public position stopped predicting private, and a lot of high-public teams slid on the final reveal.</p>
<p>To not repeat it here: trust a hold-out that matches the real test over the public board (a spatial / field-grouped split), check that a gain clears the seed band before you chase it, and don't refork on a fractional public delta. §5 of the notebook lays out both traps (CV→LB mirage + seed/refork variance) with the numbers.</p>

#### コメント 3.2 — Georgy Mamarin

- 投稿日時: 2026-06-22 17:07:13.790000
- 投票数: -3
- コメントID: `3478411`

<p>Thanks. The seed-noise point is the second trap in the notebook, so good to see it from the top of the board.</p>
<p>The ~5ft single model is what interests me, because it sits below the per-well line oracle I show (~6.6 ft pooled on train), so it reaches the wiggle, not just the offset and slope. Two questions, since they're the crux:</p>
<ol>
<li>Is ~5ft pooled row-RMSE or a per-well average? They differ by ~30% here.</li>
<li>Does it use anything beyond the well and its typewell — offset wells, spatial structure, external tops?</li>
</ol>
<p>I ask because my wall held out whole fields. If the hidden-test wells sit inside the train fields, within-field spatial interpolation is a legal signal my split throws away by design — a hole in my framing rather than a property of the data. If a single model reaches ~5ft legally, I'd like to run the harness on its OOF. That result is more interesting than the wall.</p>

##### コメント 3.2.1 — Tucker Arrants

- 投稿日時: 2026-06-22 17:31:38.893000
- 投票数: 5
- コメントID: `3478430`

<p>1) ~5ft on pooled per-row RMSE. </p>
<p>2) Only uses per well data (no cross well / spatial structure / external tops). </p>

##### コメント 3.2.2 — Unknown

- 投稿日時: 2026-06-23 02:02:33.173000
- 投票数: 0
- コメントID: `3478737`

_本文なし_

##### コメント 3.2.3 — Georgy Mamarin

- 投稿日時: 2026-06-23 02:07:25.533000
- 投票数: -2
- コメントID: `3478739`

<p>Thanks — that's the concrete answer I was hoping for, so I went and tried to reproduce it honestly.</p>
<p>I set up a per-well holdout (train on some wells, predict the hidden tail of others) and threw my best per-well models at it: a multi-scale GR particle filter, plus a CNN and a GBM with typewell-match features. The particle filter tops out around 10 ft pooled on held-out wells; the learned models did worse, and their residuals correlate ~0.6 with the PF, so blending barely helps. I couldn't get near 5 honestly.</p>
<p>Two readings and I can't tell which. Either your model does something mine doesn't, or the public test — which looks to me like copies of the train wells — is generous to everyone's LB. I can't rule out the second, so I weight my held-out CV, and that's where the ~10 comes from.</p>
<p>If your ~5 is a well-grouped held-out CV rather than the LB, it beats the wall and I'd genuinely like to know what a single model does that a particle filter doesn't. If it's the LB, the wall holds and the test is just kind to all of us. Either way I'd rather know than guess.</p>

##### コメント 3.2.4 — Shrey Gandhi

- 投稿日時: 2026-06-23 10:01:20.063000
- 投票数: -3
- コメントID: `3479050`

<p>Very interesting converstation.</p>

##### コメント 3.2.5 — Georgy Mamarin

- 投稿日時: 2026-06-23 12:16:03.287000
- 投票数: -2
- コメントID: `3479132`

<p>Walking back what I first wrote here. I thought the public board scored on three train-copy wells. It doesn't: the data page is clear those are example wells, replaced by the ~200 hidden test wells at scoring (Ioannis M caught it). So there's no easy 3-well board behind your ~5. On an honest within-field hold-out my best per-well tops out ~10 and I couldn't reach ~5, so I'd read your ~5 as a genuinely better model than mine, not a board artifact. Still keen to see what it does that a particle filter can't, if you ever share the OOF.</p>

##### コメント 3.2.6 — Ioannis M

- 投稿日時: 2026-06-23 12:38:35.530000
- 投票数: 1
- コメントID: `3479141`

<p>First of all, thanks for your report and the insights you are sharing.
Haven't gone through details but I think there is a misunderstanding: </p>
<blockquote>
  <ol>
  <li>The public test is three train wells</li>
  </ol>
  <p>…</p>
  <p>So the public scoring set is these three train wells; </p>
</blockquote>
<p>the public test set is not 3 wells, it is 26% of approx 200 test wells (as mentioned in the Overview page) </p>

##### コメント 3.2.7 — tennogh

- 投稿日時: 2026-06-23 12:54:13.870000
- 投票数: 0
- コメントID: `3479153`

<p>It's just what Claude diagnoses if you let it run freely (same with the 7ft wall). <a href="https://www.kaggle.com/tuckerarrants">@tuckerarrants</a> already clarified that he was talking about 5ft CV.</p>

##### コメント 3.2.8 — Georgy Mamarin

- 投稿日時: 2026-06-23 15:17:02.493000
- 投票数: -1
- コメントID: `3479243`

<p>You're right, thank you — I conflated the example test/ folder (those few train-derived wells) with the actual scoring set. The data page is explicit that the examples get replaced by the ~200 hidden test wells at rerun, so "the public board is three train wells" is just wrong, and I'm retracting it. What survives is the train-only side — the honest within-field CV (~10 ft for my methods) and the per-well drift not coming out of the legal features I tried; the leak explanation for the CV-vs-board gap doesn't hold. Appreciate the catch.</p>

##### コメント 3.2.9 — Georgy Mamarin

- 投稿日時: 2026-06-23 15:21:10.693000
- 投票数: -1
- コメントID: `3479250`

<p>Fair hit 😄 — though if you're going to pin it on a model running free, at least guess GPT. The real story is more mundane: I got attached to a nice-looking find and didn't re-read the data description carefully — the example test/ folder is train-derived, and I ran with "the board is leaky" instead of noticing it gets swapped for the real hidden test at scoring. A good reminder to re-read the rules now and then, not just skim them on day one. The train-only parts (oracle ladder, the ~10 honest CV) hold up; the leak framing doesn't, and Tucker's ~5 is his CV, which I'll take at face value. Fixing it now.</p>

##### コメント 3.2.10 — Unknown

- 投稿日時: 2026-06-24 13:00:21.527000
- 投票数: 0
- コメントID: `3480083`

_本文なし_

#### コメント 3.3 — Cody_Null

- 投稿日時: 2026-06-23 15:30:36.050000
- 投票数: 3
- コメントID: `3479265`

<p>Yeah, seems kaggle is ignoring the problem - at least publicly. <a href="https://www.kaggle.com/competitions/birdclef-2026/discussion/704413">https://www.kaggle.com/competitions/birdclef-2026/discussion/704413</a> I have brought this up many times at this point. No comments from admins despite fair traction. </p>

##### コメント 3.3.1 — Georgy Mamarin

- 投稿日時: 2026-06-23 17:29:04.113000
- 投票数: -2
- コメントID: `3479351`

<p>You've been carrying this one longer than most. The BirdCLEF and Santa examples in your thread make it hard to argue with. Fork-farming and the seed-noise reshuffle are the same coin: when a few-thousandths public delta gets forked a thousand times, the board stops ranking method and starts ranking luck. That's the trap I tried to make measurable here — §5 shows a "better" fork usually sits inside the seed band, so the rank movement is mostly noise. It won't fix the farming, but a hold-out that tells you your gain isn't real is one less reason to fork blindly. The platform side is the harder problem, and the silence on it is the frustrating part — you're right to keep raising it.</p>

### コメント 4 — wharekawa

- 投稿日時: 2026-06-30 10:50:29.423000
- 投票数: -1
- コメントID: `3484410`

<p>Thanks for the v2 -- the leak retraction caught us too. We'd independently concluded the public board was scored on the three visible example wells, and your correction -- example set != scoring set, ~200 hidden wells, the private rescoring -- straightened us out. Appreciated.</p>
<p>For what it's worth, our investigation converged on the same corrected picture from a different angle: a from-scratch harness reproduction (we reproduce your 9.07 / 6.59 / 3.00 and the 15.9 null exactly), and the same read that the ~9-10 band is method quality rather than a wall -- the heads at ~5.6 settle it. Convergent confirmation, not just agreement.</p>
<p>The one thing we'd add, though it may already be implicit in your section 6: a mechanistic read on why cross-well ANCC reconstruction fails leave-field-out. The error we measured (20-160 ft) tracks dip-rate x inter-well spacing -- ~0.035 ft/ft x ~1-5 kft  ~35-175 ft -- so it reads as a survey-density limit, not a method one. Your identity in section 3 (TVT = ANCC - Z + b_well) is what makes that clean: with b_well anchor-pinned, the residual is the differential dip the neighbours can't resolve. Thanks again.</p>

#### コメント 4.1 — Georgy Mamarin

- 投稿日時: 2026-07-02 06:01:47.160000
- 投票数: -3
- コメントID: `3486068`

<p>Thank you for running it rather than taking my word for it — a reproduction is the strongest check a harness can get, and it's the first one this notebook has had.</p>
<p>On the ladder: 9.07 / 6.59 / 3.00 is exactly what the published config prints (SEED=42, the 250-well subsample), so your run matches to the last digit. One config question, mostly as a handshake: which config was your flat rung? The 250-well subsample gives ~16.6, and the full 773 gives 15.91 with the ladder moving slightly (9.04 / 6.70 / 3.05 on my machine). If your 15.9 is a full-773 run, you've also done the robustness check §10 asks for — even better. And on the leak: you reaching the same wrong conclusion independently is oddly reassuring — it was a well-built trap, and I'm glad the retraction closed it for both of us.</p>
<p>Your survey-density read holds up on my data, with one refinement. On a separate recompute (not a cell in the notebook), using the notebook's own field definition: same-field nearest-neighbour spacing is median ~490 ft (p90 ~1.8 kft), and the surface-level difference between those neighbours is median ~13 ft, p90 ~79 ft — an implied gradient of median ~0.023 ft/ft, with your ~0.035 well inside the spread (p90 ~0.075). So at real spacing the law puts the reconstruction floor at roughly 13–80 ft, which is right where my IDW result (~23 ft) sits — and the old 13–129 band, which the current version recasts as an accidental same-field reconstruction, fits the same law once spacing and gradient run to their tails. Two scope notes: the plane-fit blow-ups (hundreds to thousands of ft) are ill-conditioning, not geology, so I wouldn't stretch the law to cover them; and your 1–5 kft spacing reads wider than same-field nearest-neighbour — was your neighbour set cross-field, or K-nearest rather than first-nearest? Inflated neighbour distance was exactly the trap my own v1 fell into, from the other side.</p>
<p>This is going into the notebook: §6 gets the mechanism paragraph with credit to you, and §2 a note that the ladder has been reproduced.</p>

#### コメント 4.2 — Georgy Mamarin

- 投稿日時: 2026-07-03 10:15:12.093000
- 投票数: -2
- コメントID: `3487050`

<p>Delivered: v13 is live with your mechanism in §6 (credited, with my spacing/gradient recompute alongside) and the reproduction note in §2. The flat-rung question upthread still stands when you get a chance — 16.6 would mean the 250-well config, 15.9 the full 773. Thanks again; §6 is better for it.</p>

##### コメント 4.2.1 — wharekawa

- 投稿日時: 2026-07-05 06:55:19.710000
- 投票数: -1
- コメントID: `3488690`

<p>The 15.9 was the full-773 null, so it's the §10 config, and I just re-ran to confirm end to end: on all 773, constant/line/smooth = 9.04 / 6.70 / 3.05 and carry-last 15.91 — matches your machine to 2 dp. The 9.07 / 6.59 / 3.00 in my earlier note was the 250-well subsample ladder; flagging that those two came from different configs. Harness reproduces faithfully at full scale.</p>

##### コメント 4.2.2 — Georgy Mamarin

- 投稿日時: 2026-07-05 17:17:11.027000
- 投票数: -1
- コメントID: `3489489`

<p>That closes it cleanly — thank you for re-running end to end. So the record is unambiguous: the 250-well subsample is 9.07 / 6.59 / 3.00 with flat 16.6, and the full 773 is 9.04 / 6.70 / 3.05 with flat 15.91, both from the same harness. Matching at full scale to 2 dp is exactly the independent confirmation the ladder needed, and it's yours.</p>

### コメント 5 — Georgy Mamarin

- 投稿日時: 2026-06-25 20:09:54.793000
- 投票数: -1
- コメントID: `3481297`

<p>Update: a couple of things moved since the rewrite, both in the notebook (§5, v7).</p>
<p>I tested the bimodal-tail hedge pilkwang proposed in the comments: on a worst-decile well, read the mode weight from the two GR-misfit minima and predict the posterior mean instead of committing. It helps where it should. On the ambiguous tail it cuts the datum error from ~5.8 ft (committing) to ~5.1, beating the fixed midpoint.</p>
<p>The bigger thing: the part I'd called the legal wall turned out not to be one. The datum scan (centred on the true shape, so it isolates the datum) only localizes 8% of the time under a flat-surface calibration, but fit the gain/offset on the known heel instead and it recovers ~80%, essentially the oracle's 82%. The 8% was the calibration, not the data, and Tabish's GR-rotation denoise nudges it to 84%. So the datum is legally pinnable; what's left hard is the per-well shape, not the datum.</p>
<p>The within-field hold-out is still the open piece, the second and less-pessimistic floor bracket I want. If anyone has one shaped like the real test, I'll run the harness on it.</p>

### コメント 6 — lucian kucera

- 投稿日時: 2026-06-22 12:37:52.547000
- 投票数: 0
- コメントID: `3478187`

<p>The simple reason why public plateus at 7.2 is that most of the solutions are not good, they just randomly blend others notebooks without trrying anything new.</p>

#### コメント 6.1 — Georgy Mamarin

- 投稿日時: 2026-06-22 17:30:48.557000
- 投票数: -1
- コメントID: `3478425`

<p>Agreed that a lot of the cluster is blends of blends — that's the seed/refork trap in §5, and §6 says re-forking just keeps you there. One nuance: the notebook isn't claiming 7.2 is a fundamental wall. It's a measured negative result on one specific test — can a gradient-boosted model predict the per-well drift slope on held-out fields? No. Tucker's ~5ft is the "something new" you mention; whether that's a legal signal I under-modeled or info from outside the released data is the open question.</p>

##### コメント 6.1.1 — lucian kucera

- 投稿日時: 2026-06-22 18:55:26.530000
- 投票数: 0
- コメントID: `3478513`

<p>I think i have an idea how to get 5ms i will try that this week, might get even better. But not promossing anything.</p>
