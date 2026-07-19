# Submission timeout after successful Notebook run

- 投稿者: wqi876
- 投稿日時: 2026-05-29 07:03:52.332000
- 投票数: 0
- コメント数: 1（取得数: 1）
- トピックID: `703208`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/703208](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/703208)

## 本文

<p>Hi everyone,</p>
<p>I'm experiencing a persistent timeout issue when submitting to this competition.
After uploading submission.csv and clicking "Submit", the status shows "Scoring…"
It stays like this for several hours, then eventually shows "Timeout"
I've tried submitting the same file many times at different hours, all resulted in timeout
Thank you!</p>

## コメント

### コメント 1 — PatrickAIForFun

- 投稿日時: 2026-05-29 07:13:23.463000
- 投票数: 0
- コメントID: `3464216`

<p>This is a code submission - you cannot simple upload a submission.csv file. You have to submit your inference code notebook which is then fully rerun on new (and larger) test data. Thus if your code also includes training, this is also rerun and thus might take longer than the 9h maximum.</p>
