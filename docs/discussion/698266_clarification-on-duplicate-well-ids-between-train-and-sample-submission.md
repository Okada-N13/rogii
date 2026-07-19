# Clarification on duplicate well IDs between train and sample_submission

- 投稿者: Kristof Anderson
- 投稿日時: 2026-05-09 03:42:44.614000
- 投票数: 2
- コメント数: 1（取得数: 1）
- トピックID: `698266`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698266](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698266)

## 本文

<p>Good night, organizers,
While inspecting the competition data, I noticed that some well IDs appearing in sample_submission also appear in the provided train files, and the corresponding train files appear to contain TVT values for rows that are targets in the submission.
Are participants allowed to use TVT values from provided train files directly when the same well/row IDs appear in the submission set, or should those overlapping target values be treated as unavailable/leakage?
I want to make sure my approach follows the intended rules and spirit of the competition.
Thanks.</p>

## コメント

### コメント 1 — Chris Deotte

- 投稿日時: 2026-05-09 08:21:52.967000
- 投票数: 5
- コメントID: `3455366`

<p>Hi. Read discussion <a href="https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697507">here</a>. What we see isn't the real sample submission nor real test data. When we submit our code, the sample submission and test data gets replaced and our code sees the real ones.</p>
