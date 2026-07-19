# How much should we trust the LB score?

- 投稿者: 寿!
- 投稿日時: 2026-06-04 04:06:20.449000
- 投票数: 18
- コメント数: 6（取得数: 6）
- トピックID: `704273`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/704273](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/704273)

## 本文

<p>I'm relatively new to this competition, but it seems there's some distribution shift between train and test. My local CV and public RMSE diverge by up to 2 in some cases. On top of that, which one is more optimistic (local vs. public) varies depending on the prediction approach. Methods that rely on specific assumptions or modeling tend to show a larger gap. In my case, a spatial method using offset wells gives CV < LB (public is more pessimistic), while the particle filter approach that's been popular in recent notebooks gives CV > LB (public is more optimistic). <br>
My thinking is that since the training set has 773 wells and the public test set only has 52, we should trust local CV over the LB, assuming our validation strategy is sound. <br>
What do you all think?</p>

## コメント

### コメント 1 — Ulrich G.

- 投稿日時: 2026-06-04 09:01:19.163000
- 投票数: 1
- コメントID: `3466561`

<p>I think we could trust, for the time being there is a line-up between CV and LB for me</p>

#### コメント 1.1 — 寿!

- 投稿日時: 2026-06-04 12:11:39.080000
- 投票数: 1
- コメントID: `3466640`

<p>That makes sense. I also feel like there tends to be a trend where LB improves when CV improves, though the ranges seem to be on different scales.</p>

##### コメント 1.1.1 — Tucker Arrants

- 投稿日時: 2026-06-04 15:50:34.567000
- 投票数: 0
- コメントID: `3466722`

<p>Yes, when I make larger pipeline changes, my CV-LB correlation "resets."</p>
<p>LB is quite noisy…trust your CV</p>

### コメント 2 — Jack

- 投稿日時: 2026-06-04 22:08:41.320000
- 投票数: 2
- コメントID: `3466904`

<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F21922722%2F780d4fa058448626816e8d6a0864ab3e%2Fdfa08573-5a0e-4345-bab0-b752fa2e3dc5.png?generation=1780610842567617&alt=media" alt=""></p>

### コメント 3 — Tim Krige

- 投稿日時: 2026-06-04 11:13:10.277000
- 投票数: 2
- コメントID: `3466624`

<p>In my opinion, both are important. I think that leaderboard probing is a real risk here, and your comment of trusting local CV therefore has some merit, however, dataleaks are of critical importance. I think an honest take would be to consider estimated confidence scores based on the population size of the test set. I.e., determine the confidence of the local CV being similar or different to the LB score with something like a null hypothesis test. This may help to determine if the lb score is true or fake, but requires honest engineering of the model too.</p>
<p>Hope this helps! </p>

#### コメント 3.1 — 寿!

- 投稿日時: 2026-06-04 12:15:39.953000
- 投票数: 1
- コメントID: `3466643`

<p>Thank you for the insightful advice! You're right that a statistically-grounded approach seems to be key here.</p>
