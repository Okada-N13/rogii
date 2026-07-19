# GPU vs TPU

- 投稿者: parijit
- 投稿日時: 2026-05-31 22:38:12.990000
- 投票数: 2
- コメント数: 1（取得数: 1）
- トピックID: `703519`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/703519](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/703519)

## 本文

<p>In the description its mentioned about GPU and CPU timing of notebooks. Are we allowed to use TPU as well. I do have the option of using 5ve-8 ? </p>

## コメント

### コメント 1 — Chris Deotte

- 投稿日時: 2026-05-31 23:20:18.240000
- 投票数: 5
- コメントID: `3464938`

<p>You can use whatever accelerator you want to train models and save their model weights. (You can also train and save models offline, i.e. not using Kaggle). But when you make a submission notebook, we load our saved model weights, and then we must infer on either GPU or CPU.</p>
