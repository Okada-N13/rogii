# Notebook Timeout run

- 投稿者: Savas Papadopoulos
- 投稿日時: 2026-06-02 12:08:56.845000
- 投票数: 3
- コメント数: 2（取得数: 2）
- トピックID: `703867`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/703867](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/703867)

## 本文

<p>Why does my notebook run out of time if it runs for only a few minutes? </p>

## コメント

### コメント 1 — Chris Deotte

- 投稿日時: 2026-06-02 12:40:26.413000
- 投票数: 0
- コメントID: `3465593`

<p>When we commit our notebook it only infers the 3 fake test wells. When we submit our notebook, the fake wells get replaced with 200 real test wells and our code sees and infers the 200 real test wells. </p>
<p>If you want to estimate how long your code takes (during a submit), then have your code infer 200 train wells as an example.</p>

#### コメント 1.1 — Savas Papadopoulos

- 投稿日時: 2026-06-03 10:45:11.307000
- 投票数: 1
- コメントID: `3466036`

<p>Thank you so much, Chris. Regards from Athens, Greece, Savas</p>
