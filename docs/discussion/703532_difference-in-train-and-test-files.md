# Difference in train and test files

- 投稿者: Navya Bhat
- 投稿日時: 2026-06-01 02:03:21.316000
- 投票数: 1
- コメント数: 2（取得数: 2）
- トピックID: `703532`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/703532](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/703532)

## 本文

<p>Hi organizers / fellow competitors,
I'd like to confirm a structural difference I'm seeing between the train and test files before I build my feature pipeline.
 In the training horizontal-well files, the columns are:
 MD, X, Y, Z, ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA, TVT, GR, TVT_input
 In the test horizontal-well files, the columns are only:
 MD, X, Y, Z, GR, TVT_input
 So the target TVT (expected to be hidden) and the six formation-top columns : ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA appear in train but are absent from
 test.
 My questions:</p>
<ol>
<li>Is it correct that these 6 formation columns will not be available at inference time, and therefore should be treated as train-only (i.e., not usable
as model input features)?</li>
<li>Or is their absence specific to the provided sample test files, and the actual hidden test set will include them?
I want to make sure I'm not engineering features on columns that won't exist when the notebook is scored. Confirmation would help avoid a train/test
mismatch.
Thanks!</li>
</ol>

## コメント

### コメント 1 — Tucker Arrants

- 投稿日時: 2026-06-01 02:08:44.097000
- 投票数: 1
- コメントID: `3464964`

<p>Correct, they are training only</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4379159%2F6815f6dd03b57a39d2db4d4889091b93%2FScreenshot%202026-05-31%20220830.png?generation=1780279723236619&alt=media" alt=""></p>

### コメント 2 — Unknown

- 投稿日時: 2026-06-01 09:21:07.723000
- 投票数: -1
- コメントID: `3465065`

_本文なし_
