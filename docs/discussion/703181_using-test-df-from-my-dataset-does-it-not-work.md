# Using test_df from my dataset, does it not work??

- 投稿者: Moonimonster
- 投稿日時: 2026-05-29 01:57:22.762000
- 投票数: 2
- コメント数: 2（取得数: 2）
- トピックID: `703181`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/703181](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/703181)

## 本文

<p>Hello, hope y'all enjoying this competition!!
I have small question for those of you having fun with this competition.
I am currently using my train_df, and test_df that are all preprocessed, stored in my dataset.
For some reason, my kaggle notebook works well but when i submit my submission file to score, it crashes with errors saying 'Submission score error', or 'Notebook threw exception'.
I then checked my submission file whether it had different numbers of rows, id, and so on.. but turns out it has really not much of difference from the sample_submission or the ones that was scored successfully.</p>
<p>So  I was wondering if using test_df from my dataset, not from /kaggle/input/competitions/rogii-wellbore-geology-prediction/test causes trouble.
If anyone knows about it, plz help..! I'm also currently dealing with this issue. so i'll keep updating</p>

## コメント

### コメント 1 — Chris Deotte

- 投稿日時: 2026-05-29 02:31:04.683000
- 投票数: 3
- コメントID: `3464162`

<p>We cannot use our local test_df. When we submit our code, all the test data gets replaced with the real test data. This is done so that we cannot look at the test data with our human eyes. Only our submitted code gets access to the real test data. Because of this, our submit code must process the real test data during submit (i.e. we cannot do it ahead of time).</p>

#### コメント 1.1 — Moonimonster

- 投稿日時: 2026-05-29 03:04:56.873000
- 投票数: 0
- コメントID: `3464168`

<p>Yeah that really makes sense!
Thank you!!</p>
