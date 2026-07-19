# Is online learning / test-time fine-tuning allowed?

- 投稿者: Kh0a
- 投稿日時: 2026-05-08 04:09:03.521000
- 投票数: 25
- コメント数: 5（取得数: 5）
- トピックID: `698002`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698002](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698002)

## 本文

<p>With the current evaluation setup, participants can train the model offline and then fine-tune it using the test dataset at submission time by peeking at the actual test dataset. They can only extract the available TVT_input (calibration reference) and other features but can make a solid improvement through domain adaptation.</p>
<p>Specifically, at submission time in a Kaggle notebook:</p>
<ol>
<li>Load the hidden test data (wellbores + GR, formation parameters, TVT_input)</li>
<li>Fine-tune the pre-trained model using test features as input (self-supervised learning using TVT_input as calibration targets)</li>
<li>Generate final predictions on the adapted model</li>
</ol>
<p>Is this approach considered allowed under the competition rules? Looking forward to feedback from organizers.</p>

## コメント

### コメント 1 — hengck23

- 投稿日時: 2026-05-16 17:55:59.460000
- 投票数: 1
- コメントID: `3458816`

<p>it should be allowed. Such methods had been used in previous kaggle competitions before.<br>
it also includes things like</p>
<ul>
<li>finding statistic  like mean, std of test data  </li>
<li>creating multiple window templates for matching  </li>
<li>self-supervsied learning or fine-tuning or adaptation etc  </li>
</ul>

### コメント 2 — Tucker Arrants

- 投稿日時: 2026-05-14 20:37:59.090000
- 投票数: 2
- コメントID: `3458029`

<p>Thank you for sharing your augmentation + online training technique. I consistently get about a 0.15 - 0.2 improvement from it. </p>
<p>Most recent training run:</p>
<p>5 fold LGB without -> 9.812</p>
<p>5 fold LGB, n_aug_splits = 1, online_training = True -> 9.649</p>
<p>5 fold CatBoost without -> 9.869</p>
<p>5 fold CatBoost, n_aug_splits = 1, online_training = True -> 9.713</p>

#### コメント 2.1 — Kh0a

- 投稿日時: 2026-05-15 02:01:06.993000
- 投票数: 1
- コメントID: `3458074`

<p>Well done, although i am not sure if this is allowed yet. </p>

### コメント 3 — Kh0a

- 投稿日時: 2026-05-08 15:49:55.707000
- 投票数: 2
- コメントID: `3455094`

<p>I have tested with same training setup:</p>
<p><a href="https://www.kaggle.com/code/llkh0a/rogii-lgbm-aug-online-training">online training</a>: 10.953</p>
<p><a href="https://www.kaggle.com/code/llkh0a/rogii-lgbm-aug">no online training</a>: 11.323 </p>
<p>The features processing idea was from <a href="https://www.kaggle.com/code/shinyanagai123/triple-signal-beam-search-dual-pf-lightgbm">https://www.kaggle.com/code/shinyanagai123/triple-signal-beam-search-dual-pf-lightgbm</a></p>

### コメント 4 — PC Jimmmy

- 投稿日時: 2026-05-16 15:42:41.920000
- 投票数: 0
- コメントID: `3458763`

<p>Test time learning has been an acceptable method for all the years I have been on kaggle.  The only issue I ever had was the compute time limit - in some past competitions my methods were too slow for the compute/time that was available.</p>
