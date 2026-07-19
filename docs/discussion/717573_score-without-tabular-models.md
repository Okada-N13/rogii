# Score Without Tabular Models

- 投稿者: k256.dev
- 投稿日時: 2026-07-02 06:50:42.033000
- 投票数: 17
- コメント数: 19（取得数: 19）
- トピックID: `717573`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/717573](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/717573)

## 本文

<p>Tabular models can definitely be considered a powerful approach in this competition. They are used as the base for many of the top public notebook solutions, and my own best score was also achieved using a tabular model. Since I have not yet done a large-scale ensemble, I think there is still room to improve my current approach by about 0.2 points.</p>
<p>However, I also feel that this approach may eventually hit a ceiling. So, if possible, I would like to know what realistic scores others are getting without using tabular models.</p>
<p>Now that the public notebooks are full of duplicated solutions, I hope this discussion helps convey that the methodology itself, and improving accuracy through better methods, are what truly matter.</p>
<p>That said, the value and the interesting part of this competition lie precisely in the “method,” so it is probably wiser not to disclose too many details about it.</p>
<p>My best score without using a tabular model is <strong>7.098</strong>, achieved with a single method. To clarify, my current best submission score is <strong>6.798</strong>, which was achieved with a tabular model and does not incorporate this non-tabular method at all.</p>

## コメント

### コメント 1 — Angus Chang

- 投稿日時: 2026-07-17 17:55:06.567000
- 投票数: 6
- コメントID: `3499632`

<p>Joined this competition a week ago and just made my first sub<br>
It’s a single pure physics-based model: CV 6.85 / LB 6.577</p>

#### コメント 1.1 — Poobear

- 投稿日時: 2026-07-17 23:37:42.997000
- 投票数: 0
- コメントID: `3499791`

<p>Congratulations on the strong first result. Which observation does the 6.577 pure-physics model assimilate: horizontal/typewell GR, trajectory geometry, other query wells/spatial structure, or another provided field? Does the 6.85 CV group by whole well and simulate the hidden suffix?</p>

##### コメント 1.1.1 — Angus Chang

- 投稿日時: 2026-07-18 08:47:33.780000
- 投票数: 1
- コメントID: `3499918`

<p>Thanks but I'd rather keep the model detail to myself for now:)</p>
<blockquote>
  <p>Does the 6.85 CV group by whole well and simulate the hidden suffix?  </p>
</blockquote>
<p>Followed the competition metric exactly (per-point RMSE on the post PS part, not the per-well avg)</p>

### コメント 2 — radiant-allomancer

- 投稿日時: 2026-07-13 14:50:54.893000
- 投票数: -9
- コメントID: `3496286`

<p><a href="https://www.kaggle.com/k256.dev">@k256.dev</a> Your comments say ordinary 5-fold LightGBM/CatBoost/XGBoost, no target change or feature engineering, with the input features doing the work. Without asking for exact columns: are those inputs mainly (a) raw or row-local horizontal-well columns, (b) features derived from paired typewell/GR alignment, or (c) predictions/candidate paths from another model? Even the broad category would help reproduce the formulation. Thanks.</p>

### コメント 3 — radiant-allomancer

- 投稿日時: 2026-07-13 12:45:31.207000
- 投票数: -10
- コメントID: `3496223`

<p><a href="https://www.kaggle.com/shu01">@shu01</a> Congratulations on the 4.859 lead. If you're willing to share only at a coarse level: is your current system mainly per-row tabular, per-well sequential/non-tabular, or an ensemble of both? Does each prediction use only that well's legal inputs, or any cross-well/spatial/external information? No implementation details needed—this would help clarify which problem formulation is proving viable. Thank you.</p>

### コメント 4 — Tucker Arrants

- 投稿日時: 2026-07-02 18:00:24.197000
- 投票数: 4
- コメントID: `3486609`

<p>You can get CV in the 5s with non-tabular models. I don't know what is possible with tabular models. </p>

#### コメント 4.1 — k256.dev

- 投稿日時: 2026-07-11 15:01:36.297000
- 投票数: 3
- コメントID: `3495170`

<p>Today, both my tabular approach and my non-tabular approach finally reached a CV in the 5.x range. Their LB scores are also consistent with the CV.</p>
<p>I just hope the correlation holds from here on. I'll keep trusting the CV.</p>

##### コメント 4.1.1 — LP

- 投稿日時: 2026-07-12 08:55:30.037000
- 投票数: 0
- コメントID: `3495418`

<p>Could you please create a post to share the improvement ideas for the table model? Or feel free to share your own insights. I think the current open-source solutions really need the ideas of the table model.</p>

##### コメント 4.1.2 — k256.dev

- 投稿日時: 2026-07-12 09:28:19.823000
- 投票数: 3
- コメントID: `3495424`

<p><a href="https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/717573#3488586">https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/717573#3488586</a></p>
<p>I'm afraid I can't share any further details because my approach is probably quite different from those of the other top competitors, and revealing it would likely put my current ranking at significant risk.</p>
<p>What I can say, though, is that the models themselves are nothing special—they're simply a 5-fold × 3-model ensemble of LightGBM, CatBoost, and XGBoost.</p>
<p>One more thing I'd like to mention: in my opinion, the current best public notebook (scoring below 7.3) is focusing on improvements that are largely off target. Small incremental tweaks alone are unlikely to produce meaningful gains. Moreover, the noticeable gap between CV and LB suggests that the model itself may not be fundamentally robust. It appears to carry a substantial risk of a large shake-down, and I wouldn't be surprised if its score ended up deteriorating to around 10 on the private leaderboard.</p>

##### コメント 4.1.3 — LP

- 投稿日時: 2026-07-12 10:36:38.880000
- 投票数: 0
- コメントID: `3495460`

<p>Thank you very much for your sharing. Keep it up and win the gold medal!</p>

### コメント 5 — Zejun_

- 投稿日時: 2026-07-02 19:14:56.530000
- 投票数: 2
- コメントID: `3486657`

<p>Tabular models with 7.0x, non-tabular models might be the future.</p>

### コメント 6 — YYH

- 投稿日時: 2026-07-12 14:46:03.880000
- 投票数: 0
- コメントID: `3495595`

<p>Has anyone tested how well a standalone physical model can perform?</p>

### コメント 7 — APIyuya

- 投稿日時: 2026-07-05 04:08:55.163000
- 投票数: 0
- コメントID: `3488559`

<p>Reading your thoughts on how "non-tabular models might be the future," alongside Tucker’s comment about reaching CV in the 5s with them, honestly leaves me feeling completely devastated right now.
The reason is that we have been desperately trying to break through the 7.15 wall by heavily investing in non-tabular approaches—specifically 1D-CNNs and Cross-Attention architectures. However, our deep learning journey was an absolute trainwreck. No matter what we tried, the models suffered from severe "confidence jumps" due to the self-similarity of the GR logs (the bimodal ambiguity issue), failing to yield any consistent improvements over a simple flat baseline.
Having completely hit a dead end with non-tabular models, it is utterly mind-boggling to us how you managed to achieve such an incredible score of 6.798 using a tabular model. Since many public solutions seem to saturate around 7.15, what kind of breakthrough allowed you to stretch the capability of a tabular framework that far?
Without disclosing too many proprietary details, could you give us a hint on the direction of your breakthrough?</p>
<ol>
<li>Is it about the "geological quality" of feature engineering?
Did you find a way to elegantly inject the "stratigraphic matching quality"—which we failed to capture via CNN/Attention layers—directly into your GBDT feature factory?</li>
<li>Or is it about redefining the target itself?
You mentioned that you haven't done a large-scale ensemble yet, which means your single/few tabular models are doing the heavy lifting. Did you change what the model is actually predicting (e.g., predicting global parameters like slope/offset instead of a simple per-row TVT regression)?
Having blown ourselves up trying to make Deep Learning work, the sheer magic behind your high-scoring tabular approach is both deeply puzzling and inspiring. We would love to hear your high-level philosophy on how you conquered this wall!</li>
</ol>

#### コメント 7.1 — k256.dev

- 投稿日時: 2026-07-05 05:00:34.077000
- 投票数: 9
- コメントID: `3488586`

<p>Today's experiments showed that tabular models can achieve even better results. This outcome is different from what I initially believed, and honestly, I'm a bit surprised myself.</p>
<ol>
<li><p><strong>I didn't use feature engineering.</strong> I tried Featuretools once, but it didn't improve the performance. I also didn't use any unusual models—just a 5-fold ensemble of LightGBM, CatBoost, and XGBoost. What really matters is the input features themselves.</p></li>
<li><p><strong>I didn't change the prediction target.</strong> However, I do think that representing the target in a more appropriate way could be beneficial. Another possibility is a common idea from financial forecasting: predicting an auxiliary future-related target and then using those predictions as features for the main model. A similar approach might also be effective for this competition.</p></li>
</ol>
<p>I think the most important things are to study the data yourself and pay close attention to the discussions. I'm confident that our current results came from continuously improving the existing tabular notebook, rather than introducing an entirely different modeling framework.</p>
<p>I also don't think LLMs are particularly good at generating the key research ideas. The improvements we've achieved so far came primarily from my own ideas. I find LLMs most useful for organizing my thoughts, implementing ideas, or helping with research and information gathering, rather than for coming up with the core breakthroughs themselves.</p>

##### コメント 7.1.1 — APIyuya

- 投稿日時: 2026-07-05 05:23:13.443000
- 投票数: -3
- コメントID: `3488601`

<p>Wow, thank you so much for such an incredibly generous and eye-opening reply! I am genuinely blown away by your insights.
Your comment about focusing on the "input features themselves" rather than over-engineering them, and simply maximizing the standard ensemble (LGBM/CatBoost/XGBoost), completely realigns my perspective. It is deeply reassuring to know that the breakthrough didn't come from some exotic, over-complicated architecture, but from rigorous data analysis and iterative improvements on the existing tabular framework.
Also, your hint about "predicting an auxiliary target or future-related quantities"—similar to financial forecasting—is absolutely brilliant. That gives us a beautiful, concrete direction to rethink how we structure our tabular inputs and targets without breaking the core GBDT setup.
I also completely agree with your take on LLMs. They are great for structuring thoughts and speeding up implementation, but the true core breakthroughs only come from staring at the data and relying on our own human intuition.
Thank you again for sharing your high-level philosophy and giving me the clarity I desperately needed.</p>

##### コメント 7.1.2 — LP

- 投稿日時: 2026-07-10 02:35:20.160000
- 投票数: 0
- コメントID: `3494461`

<p>What is the current upper limit of tabular models, and compared to publicly available high-scoring notebooks, what is the biggest potential breakthrough?</p>

##### コメント 7.1.3 — k256.dev

- 投稿日時: 2026-07-10 02:51:23.983000
- 投票数: 1
- コメントID: `3494466`

<p>LB6.0↓ (CV7.0)</p>

##### コメント 7.1.4 — k256.dev

- 投稿日時: 2026-07-10 06:10:43.973000
- 投票数: 0
- コメントID: `3494512`

<p>Just to clarify, I'm not saying that tabular models are inherently powerful.</p>
<p>Even without using tabular models, I've been able to get below LB 6.0. (Keep in mind that my CV is still poor, so I don't consider this reliable yet.)</p>
<p>As several of the top public leaderboard competitors have pointed out, I don't think this is fundamentally a model selection issue.</p>

##### コメント 7.1.5 — Poobear

- 投稿日時: 2026-07-17 23:39:03.353000
- 投票数: 0
- コメントID: `3499792`

<p>One split detail would make the CV comparable: does the 5-fold CV group by whole well and recreate the organizer's TVT_input prefix/suffix mask? No feature details needed.</p>
