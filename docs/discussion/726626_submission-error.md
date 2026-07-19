# submission error

- 投稿者: Dalia Mowafy
- 投稿日時: 2026-07-15 22:04:21.555000
- 投票数: -3
- コメント数: 1（取得数: 1）
- トピックID: `726626`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/726626](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/726626)

## 本文

<p>Your notebook hit an unhandled error while rerunning your code. Note that the hidden dataset can be larger/smaller/different than the public dataset; when submitting this error occur; Is the hidden dataset is a new training and test data or only a new test data?</p>

## コメント

### コメント 1 — PatrickAIForFun

- 投稿日時: 2026-07-16 07:29:55.140000
- 投票数: 0
- コメントID: `3498932`

<p>It is only a new test set with ~200 wells (if I remember correctly); If your notebook runs fine on the given dummy test wells but fails during submission it is most likely related to some out of memory error (as there are many more & larger wells)</p>
