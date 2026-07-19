# be careful if you are using transformer and bfloat16

- 投稿者: hengck23
- 投稿日時: 2026-05-30 05:26:30.043000
- 投票数: 18
- コメント数: 1（取得数: 1）
- トピックID: `703344`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/703344](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/703344)

## 本文

<p>i am puzzled why my CNN works and transformer doesn't.  </p>
<p>in fact, for debug, i make a "copy transformer" where input seq = target tvt.
then i realise it is bfloat16 causing all the problems, even though i have already normalised input by mean,std</p>
<p>below are prediction of "copy transformer"  using bfloat16 and float32. </p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fdbf6aedcd2248f973085cae76ca11541%2FSelection_3811.png?generation=1780118773002280&alt=media" alt=""></p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F5214361810241cd82bed9b677232d495%2FSelection_3810.png?generation=1780118786902352&alt=media" alt=""></p>

## コメント

### コメント 1 — water joe

- 投稿日時: 2026-06-08 13:01:59.630000
- 投票数: 0
- コメントID: `3468108`

<p>Could you please tell me why this is the case? The performance of bfloat16 is very poor here.</p>
