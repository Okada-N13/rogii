# Does the submission need gpu quota from my account?

- 投稿者: Yield Smarter
- 投稿日時: 2026-05-27 06:39:35.659000
- 投票数: 2
- コメント数: 1（取得数: 1）
- トピックID: `702859`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/702859](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/702859)

## 本文

<p>Just a quick question, as I blew through my quota in no time with 3 submission without even noticing, or did i do something wrong?</p>

## コメント

### コメント 1 — PatrickAIForFun

- 投稿日時: 2026-05-27 07:22:27.183000
- 投票数: 0
- コメントID: `3463535`

<p>The submission scoring itself does not take GPU quota. However when submitting it does run the submission beforehand on your own quota with the dummy test set to check whether it prodcues a valid submission.csv.
One way to go around this would be to check whether there are more than 3 test cases and only then run the full code, otherwise just submit a dummy submission.</p>
