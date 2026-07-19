# Coding platform

- 投稿者: David Ubuara
- 投稿日時: 2026-05-06 12:14:27.746000
- 投票数: 2
- コメント数: 3（取得数: 3）
- トピックID: `697500`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697500](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697500)

## 本文

<p>Hi everyone, 
Please should I run my code locally on my PC or use the Kaggle Notebook for this competition. And do we have to submit the Notebook here if I run my code locally. Thank you</p>

## コメント

### コメント 1 — Aly Ayman

- 投稿日時: 2026-06-08 20:03:47.997000
- 投票数: 1
- コメントID: `3468331`

<p>Hello David,</p>
<p>I recommend Develop and train locally on your PC.</p>
<p>By this you can write clean code in our shared repo (with Git, proper testing, and experiment tracking). Local is much better for actually building things — Kaggle Notebooks are painful for collaboration and iteration.</p>
<p>And for the final submission:</p>
<ol>
<li>Train your model locally & you get a model file (e.g. model.pkl)</li>
<li>Upload that trained model to Kaggle as a private Dataset</li>
<li>Upload our src/ code as another Dataset</li>
<li>Write a small inference notebook on Kaggle that loads the model, predicts on the test data, and saves submission.csv</li>
<li>Commit And run that notebook then Submit</li>
</ol>
<p>This will help you professionaly in the experimentation phase and if you have a team you can do proper team work.</p>
<p>Wish you all the luck</p>

### コメント 2 — PC Jimmmy

- 投稿日時: 2026-05-07 01:31:29.283000
- 投票数: 1
- コメントID: `3454364`

<p>I am running all training local - I have lots of compute so it makes sense for me.  It does require that I save trained models in kaggle datasets for the prediction code.</p>
<p>The prediction and generation of the submission must be made on kaggle.</p>

#### コメント 2.1 — David Ubuara

- 投稿日時: 2026-05-07 10:07:05.687000
- 投票数: 0
- コメントID: `3454480`

<p>Thank you Jimmmy for that clarification. </p>
