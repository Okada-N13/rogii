# Handling Rolling and Lag features for Testing data

- 投稿者: Pratyaksh
- 投稿日時: 2026-06-04 10:18:13.945000
- 投票数: -1
- コメント数: 7（取得数: 7）
- トピックID: `704383`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/704383](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/704383)

## 本文

<p>I have done my model training but i am puzzled on how to create the rolling and lag features for infrence  or test data . how are you guy doing it</p>

## コメント

### コメント 1 — PatrickAIForFun

- 投稿日時: 2026-06-04 10:24:22.030000
- 投票数: 0
- コメントID: `3466608`

<p>What exactly is your issue there? The test set at inference looks exactly the same as the trainung data (except for the train-only coljmns TVT, ANCC, …, Geology). Thus, the approach you used to generate your training features does work exactly the same on the test files.</p>

#### コメント 1.1 — Pratyaksh

- 投稿日時: 2026-06-04 10:33:17.383000
- 投票数: 0
- コメントID: `3466609`

<p>Maybe I'm misunderstanding something then.</p>
<p>For example, suppose I create a <code>GR_rolling_20</code> feature. In training I can choose to drop the first 19 rows (or use <code>min_periods=1</code>), but at inference time the test well also starts at row 0 and has no prior history before that point.</p>
<p>Would you typically compute the first few rows using the available history only (e.g. row 0 uses one sample, row 1 uses two samples, etc.), or is there another convention people usually follow in this competition?</p>

##### コメント 1.1.1 — PatrickAIForFun

- 投稿日時: 2026-06-04 10:45:27.503000
- 投票数: 0
- コメントID: `3466614`

<p>You only need to predict the TVT starting at the prediction start point (the point where TVT_input becomes NaN). Thus there is always enough history before this point which you can use but do not need to predict. For a visualization also have a look at the image visualizations in the dataset - these do show this start point.</p>

##### コメント 1.1.2 — Pratyaksh

- 投稿日時: 2026-06-04 10:57:46.197000
- 投票数: 0
- コメントID: `3466617`

<p>idk if we are having a misunderstanding or something but i am talking about test dataset . and how to estimate those feature in test data for submission not even talking about the training or evaluation dataset</p>

##### コメント 1.1.3 — Chris Deotte

- 投稿日時: 2026-06-04 13:11:36.827000
- 投票数: 0
- コメントID: `3466666`

<p>I don't understand your question</p>
<blockquote>
  <p>but at inference time the test well also starts at row 0 and has no prior history before that point.</p>
</blockquote>
<p>The train data and test data are exactly the same. Whatever you do on train data you do the exact same thing on test data. (with exception that test is missing geological formations and true targets)</p>

##### コメント 1.1.4 — Pratyaksh

- 投稿日時: 2026-06-11 09:52:56.410000
- 投票数: 0
- コメントID: `3471334`

<p>wait, correct me if i am wrong . Like test data is similar to train data . So we don't need to predict TVT values for whole of test data we need to predict them after the NAN values starts? . If yes this will clear my confusion. I was thinking that we need the predict TVT for every test file from start of the row</p>

##### コメント 1.1.5 — PatrickAIForFun

- 投稿日時: 2026-06-11 10:01:05.823000
- 投票数: 0
- コメントID: `3471338`

<p>Exactly, you only need to predict tge rows where TVT_input is NaN. The rows before this predictuon start are already given.</p>
