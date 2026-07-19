# Does CV correlates with LB？

- 投稿者: yuanzhe zhou
- 投稿日時: 2026-07-04 23:17:13.797000
- 投票数: 19
- コメント数: 15（取得数: 15）
- トピックID: `719389`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/719389](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/719389)

## 本文

<p>It seems that the CV for public high scoring notebook is around 10 RMSE, but the LB for that notebook is 7. Does your CV correlates with LB? </p>
<p>LLM helped me to build a model and the CV/LB are both around 10. </p>
<p>Edit : From the discussions, I believe CV is what we should aim for firstly.</p>

## コメント

### コメント 1 — yu4u

- 投稿日時: 2026-07-07 15:41:11.643000
- 投票数: 17
- コメントID: `3493317`

<p>It seems that the conclusion has already been reached, but this competition is clearly a trust-your-CV competition.
I compute my CV as the average over 5 folds × 5 seeds, and only adopt changes that show a meaningful improvement there.
As a result, my CV and LB have been correlating quite cleanly.</p>
<table>
<thead>
<tr>
<th>CV</th>
<th>Public LB</th>
<th>gap (LB−CV)</th>
</tr>
</thead>
<tbody>
<tr>
<td>6.983</td>
<td>7.324</td>
<td>+0.341</td>
</tr>
<tr>
<td>6.855</td>
<td>7.154</td>
<td>+0.299</td>
</tr>
<tr>
<td>6.227</td>
<td>6.559</td>
<td>+0.332</td>
</tr>
<tr>
<td>6.051</td>
<td>6.334</td>
<td>+0.283</td>
</tr>
<tr>
<td>5.692</td>
<td>6.017</td>
<td>+0.325</td>
</tr>
</tbody>
</table>

#### コメント 1.1 — SpeedSci

- 投稿日時: 2026-07-07 15:46:19.487000
- 投票数: 0
- コメントID: `3493321`

<p>wow！good！This looks quite stable.</p>

#### コメント 1.2 — Jeevan Jolly

- 投稿日時: 2026-07-07 17:11:23.457000
- 投票数: 0
- コメントID: `3493378`

<p>What is your single model best CV, if you don't mind me asking.</p>

#### コメント 1.3 — tennogh

- 投稿日時: 2026-07-08 12:45:48.227000
- 投票数: 2
- コメントID: `3493756`

<p>Interesting that your CV is consistently below your LB. For me it's the opposite. Maybe my CV scheme is too hard.</p>

### コメント 2 — k256.dev

- 投稿日時: 2026-07-05 03:04:23.737000
- 投票数: 8
- コメントID: `3488531`

<p>I think that only part of the CV data is truly correlated with the LB.</p>
<p>The 773 CV wells can be divided into two groups based on a certain rule. From what I've observed, the LB correlates well with one group but shows little to no correlation with the other. If you look at the relationship between the LB and the CV, while also considering that the LB consists of very few samples and is therefore quite noisy, an interesting pattern emerges. (I'll leave the rule for others to figure out.)</p>
<p>I've repeatedly observed cases where a <strong>-0.9</strong> improvement in CV resulted in only a <strong>+0.1</strong> change on the LB. Excluding data leakage and pure luck, I believe that genuine improvements in CV are still likely to translate into improvements on the private leaderboard, even if the public LB appears to move in the opposite direction due to noise.</p>

#### コメント 2.1 — SpeedSci

- 投稿日時: 2026-07-05 07:14:54.657000
- 投票数: 0
- コメントID: `3488705`

<p>Wouldn't this kind of split cause a lot of fluctuation in the LB score?</p>

##### コメント 2.1.1 — k256.dev

- 投稿日時: 2026-07-05 08:00:41.647000
- 投票数: 1
- コメントID: `3488752`

<p>It depends on the current RMSE, but simply changing the random seed at one point in the algorithm can already cause fluctuations of around 0.2–0.3. I consider that to be a fairly large amount of variance.</p>
<p>That said, I also expect the variance not to become much larger than that, and I believe there's a specific reason for it—one that's related to the hypothesis I mentioned.</p>

### コメント 3 — Tucker Arrants

- 投稿日時: 2026-07-05 00:08:13.287000
- 投票数: 4
- コメントID: `3488480`

<p>I lost basically all correlation between CV and LB once CV starting getting below 6. Before that, it was decently correlated. </p>

#### コメント 3.1 — yuanzhe zhou

- 投稿日時: 2026-07-05 08:23:29.420000
- 投票数: 3
- コメントID: `3488782`

<p>Thanks for the info. From the information above (from different kagglers), I believe CV is very important in this competition. But LB also matters (high LB means that you have done something correct, but validate it wrongly).</p>

#### コメント 3.2 — SpeedSci

- 投稿日時: 2026-07-05 08:31:58.687000
- 投票数: 0
- コメントID: `3488796`

<p>Do you think we should deliberately handle those bad wells? And do you think Unet is worth using?</p>

#### コメント 3.3 — Rishikesh Jani

- 投稿日時: 2026-07-08 04:40:26.960000
- 投票数: 1
- コメントID: `3493589`

<p>Same here. Around the 5.8 mark. </p>

### コメント 4 — ImperfectKitto

- 投稿日時: 2026-07-05 00:21:02.577000
- 投票数: 1
- コメントID: `3488482`

<p>correlation isn't "="</p>
<p>if with lower CV you get lower LB, you're chilling. that said, I had many inversions (lower CV but higher LB, not other way around though)</p>

### コメント 5 — tennogh

- 投稿日時: 2026-07-04 23:21:37.003000
- 投票数: 1
- コメントID: `3488459`

<p>It correlates to some extent, there is a previous thread where people have shared their figures. But LB is very noisy (only ~50 wells vs 773 wells for CV). The public notebooks are probably relying a lot on seed noise and their CV is pretty high. </p>

### コメント 6 — Sasha Turutin

- 投稿日時: 2026-07-04 23:20:03.953000
- 投票数: 1
- コメントID: `3488457`

<p>In my case not very much, but maybe I'm doing something wrong. Never reached CV less than 8 so far.</p>

### コメント 7 — Georgy Mamarin

- 投稿日時: 2026-07-05 17:41:08.040000
- 投票数: -14
- コメントID: `3489528`

<p>Short version: your CV and the public LB are scored on different sets of wells, so they won't line up cleanly. The public LB is only ~50 wells (a friendlier slice of the ~200 hidden), scored with real sampling noise; a pooled by-well CV over all 773 training wells is the honest number, and flat/persistence there is ~15.9, not 7.5. So a public notebook reading CV 10 / LB 7 is mostly the LB well-set being easier plus a lucky draw, not a broken CV.</p>
<p>On whether to trust a CV gain: the thing to beat is the seed band. pilkwang posted byte-identical notebooks scoring 7.168–7.286 on the public board — ~0.12 from nothing but the reseed — so a gain smaller than that can move the LB the wrong way purely by chance (exactly the −0.9 CV / +0.1 LB inversions people are seeing). Tucker's point that correlation dies below CV ~6 fits: past that, the differences you're chasing are finer than the ~50-well LB can resolve, so they drown in the sampling noise. Trust a by-well grouped CV over the public board, and check a gain clears the seed band before chasing it.</p>
