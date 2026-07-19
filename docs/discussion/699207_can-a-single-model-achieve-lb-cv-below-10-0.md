# Can a single model achieve LB/CV below 10.0?

- 投稿者: NobelK
- 投稿日時: 2026-05-13 05:11:56.953000
- 投票数: 17
- コメント数: 11（取得数: 11）
- トピックID: `699207`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699207](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699207)

## 本文

<p>Hi everyone,</p>
<p>I would like to ask whether it seems possible to achieve a score below 10.0 on the LB or local CV using a single model.</p>
<p>At the moment, my single CatBoost model has plateaued. I am using mostly tabular features with CatBoost, and my local CV and LB are no longer improving much after basic parameter tuning. I suspect that either my validation strategy or feature engineering may be the bottleneck, but I am not sure how to diagnose it.</p>
<p>Since I am still a beginner, I would really appreciate any discussion or exchange of ideas around this topic.</p>
<p>For example, I am interested in hearing about:</p>
<ul>
<li>whether people have seen single-model scores below 10.0</li>
<li>whether CatBoost alone seems strong enough for this competition</li>
<li>which direction is more promising: feature engineering, validation design, post-processing, or ensembling</li>
<li>common mistakes that may cause a CatBoost baseline to get stuck</li>
</ul>
<p>I am not asking for anyone’s private solution, but I would be grateful for any general hints, observations, or advice that could help beginners understand where to focus next.</p>
<p>Thanks in advance!</p>

## コメント

### コメント 1 — SpeedSci

- 投稿日時: 2026-06-27 14:32:30.770000
- 投票数: 3
- コメントID: `3482616`

<p>LGB+5Fold，get 7.034</p>

### コメント 2 — Vishal Kishore

- 投稿日時: 2026-05-24 10:45:49.383000
- 投票数: 4
- コメントID: `3462642`

<p>Yeah it is possible, I scored 8.8 with a single dl model approach only. Only matters how you formulate it in your model </p>

#### コメント 2.1 — NobelK

- 投稿日時: 2026-05-24 14:26:39.577000
- 投票数: 0
- コメントID: `3462678`

<p>That's a fantastic score!</p>
<p>I have a basic question: what does it mean to "formulate it in your model"?</p>

### コメント 3 — Andrew Lukyanenko

- 投稿日時: 2026-05-20 09:54:33.780000
- 投票数: 2
- コメントID: `3461347`

<p>This is definitely possible. I got 9.463 with a single model.</p>

#### コメント 3.1 — NobelK

- 投稿日時: 2026-05-20 10:02:44.883000
- 投票数: 0
- コメントID: `3461348`

<p>9.463 with a single model!? That's amazing!
I'm very interested in your approach.
I've run out of ideas right now…</p>

### コメント 4 — Tom

- 投稿日時: 2026-05-17 08:12:29.687000
- 投票数: 1
- コメントID: `3459030`

<p>Current GBDTs are definitely not a good solution for this challenge. Based on my current EDA, there is still significant room for improvement. Ultimately, it depends on how the problem is reformulated. I'll post some new directions after I understand the data deeper.</p>

#### コメント 4.1 — hengck23

- 投稿日時: 2026-05-17 08:46:35.987000
- 投票数: 1
- コメントID: `3459042`

<p>my experment results: upper bound 3.5 (due to ambigous annotation). i think a good model is about 4.5. 
GBDTs is ok if the input is good. </p>
<p>but problem is not feature engineering, it is problem formulation. It is easier to work with CNN and transformer ( top k path query)</p>

##### コメント 4.1.1 — NobelK

- 投稿日時: 2026-05-17 08:51:32.527000
- 投票数: 1
- コメントID: `3459045`

<p>I'd like to use CNNs and transformers, but I lack the knowledge to build a model properly.</p>
<p>I would be grateful if you could share any helpful resources or resources you know of.</p>

### コメント 5 — Tucker Arrants

- 投稿日時: 2026-05-13 11:37:33.610000
- 投票数: 2
- コメントID: `3457221`

<p>Of course, the competition has only been live for a week. I have a simple LGB that scores 9.7 on the leaderboard - nothing fancy, just some feature engineering. I’m sure the final scores will be much lower, maybe around 5 feet, but it is hard to tell. Look at the leaderboard…there are some heavy hitters competing here and it has only just begun - it's going to be a good one.</p>

#### コメント 5.1 — NobelK

- 投稿日時: 2026-05-13 12:12:38.163000
- 投票数: 1
- コメントID: `3457236`

<p>9.7… That's fantastic.</p>
<p>I still seem to lack the fundamentals, so I'll do my best to catch up.</p>
<p>Thank you for the helpful information; let's both do our best.</p>

### コメント 6 — Yang Wei Hao

- 投稿日時: 2026-05-13 07:45:38.477000
- 投票数: -1
- コメントID: `3457145`

<p>i think it must do that</p>
