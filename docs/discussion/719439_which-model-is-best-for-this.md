# which model is best for this ?

- 投稿者: Rahib Vk
- 投稿日時: 2026-07-05 02:43:27.523000
- 投票数: -7
- コメント数: 10（取得数: 10）
- トピックID: `719439`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/719439](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/719439)

## 本文

<p>which model is best for this ?</p>

## コメント

### コメント 1 — yu4u

- 投稿日時: 2026-07-05 03:07:31.023000
- 投票数: 6
- コメントID: `3488532`

<p>Fable 5.  </p>

#### コメント 1.1 — ImperfectKitto

- 投稿日時: 2026-07-05 11:31:53.313000
- 投票数: -1
- コメントID: `3489006`

<p>Not a joke btw</p>

##### コメント 1.1.1 — tennogh

- 投稿日時: 2026-07-05 11:34:26.750000
- 投票数: 0
- コメントID: `3489009`

<p>I have only tried once, but it immediately reverted to Opus, do you guys actually manage to use Fable?</p>

##### コメント 1.1.2 — ImperfectKitto

- 投稿日時: 2026-07-05 11:39:34.717000
- 投票数: 0
- コメントID: `3489015`

<p>Yep. I had it running the entire night, and it broke plateau of 7 RMSE I had for weeks</p>

##### コメント 1.1.3 — Ochir Dorzhiev

- 投稿日時: 2026-07-05 12:01:56.257000
- 投票数: 0
- コメントID: `3489032`

<p>If I may ask, did it help with the tabular model or the non-tabular model?</p>

##### コメント 1.1.4 — ImperfectKitto

- 投稿日時: 2026-07-05 15:13:38.680000
- 投票数: 0
- コメントID: `3489289`

<p>It was neither</p>

##### コメント 1.1.5 — Andrey Chankin

- 投稿日時: 2026-07-05 17:39:30.253000
- 投票数: 1
- コメントID: `3489523`

<p>so, no ML at all?</p>

##### コメント 1.1.6 — ImperfectKitto

- 投稿日時: 2026-07-05 20:05:16.153000
- 投票数: -5
- コメントID: `3489736`

<p>What I meant is that Fable uploaded itself for the submission.</p>
<p>No, what it helped with wasn't a model. An ambiguous question deserves an ambiguous answer.</p>

##### コメント 1.1.7 — Andrey Chankin

- 投稿日時: 2026-07-05 23:16:57.470000
- 投票数: 0
- コメントID: `3489873`

<blockquote>
  <p>What I meant is that Fable uploaded itself for the submission.</p>
  <p>No, what it helped with wasn't a model. An ambiguous question deserves an ambiguous answer.</p>
</blockquote>
<p>I just ment that if it was neither tabular or non-tabular it leaves no space for ML-approach variation.</p>
<p>If we rephrase the original question - what kind of help it was? - architectural, new features, postrprocessing, etc?</p>
<p>If you mind sharing for sure </p>

##### コメント 1.1.8 — ImperfectKitto

- 投稿日時: 2026-07-06 00:00:33.977000
- 投票数: 2
- コメントID: `3489915`

<p>I'm not gonna say what it was exactly (partially because I don't really understand what it was), but it was a correction (refinement) of a non-ML method in my ensemble (definetelly not tabular or a CNN).</p>
<p>The most interesting part isn't what the method is anyway. I have a pretty mature codebase and documentation (for Claude) where I keep track of all experiments. It resurfaced one experiment of mine, noticed an idealogical error in it (experiment's execution didn't give an idea a fair chance), fixed it, re-executed the experiment, and submitted it (which gave -0.2 on LB). All by itself - I actually prompted it to research and implement new ideas. That error escaped Opus for like 10-20 sessions.</p>
