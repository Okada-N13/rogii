# Issues with submission format

- 投稿者: jpinzon
- 投稿日時: 2026-05-22 11:57:17.580000
- 投票数: 2
- コメント数: 3（取得数: 3）
- トピックID: `702308`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/702308](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/702308)

## 本文

<p>I am stuck at the submission format. I am not sure why. My file and data looks just like others in the competition and follows the format required in the instructions. Where can I get help for more details on the issue? I also see the file size and number of rows being the same as others. </p>

## コメント

### コメント 1 — jpinzon

- 投稿日時: 2026-05-22 11:59:40.117000
- 投票数: 1
- コメントID: `3462180`

<p>Here is the results from my submission creator : Compiling submission frame in Kaggle format…
✓ Submission shape: (14151, 2)
✓ Sample IDs: ['000d7d20_1442', '000d7d20_1443', '000d7d20_1444']</p>
<p>✓ Success! Submission saved to 'submission.csv' in correct Kaggle format</p>
<ul>
<li>Total predictions: 14151</li>
</ul>
<p>First 5 rows of submission:
shape: (5, 2)
┌───────────────┬──────────────┐
│ id            ┆ tvt          │
│ ---           ┆ ---          │
│ str           ┆ f32          │
╞═══════════════╪══════════════╡
│ 000d7d20_1442 ┆ 12051.21875  │
│ 000d7d20_1443 ┆ 12051.330078 │
│ 000d7d20_1444 ┆ 12051.441406 │
│ 000d7d20_1445 ┆ 12039.623047 │
│ 000d7d20_1446 ┆ 12040.129883</p>

#### コメント 1.1 — jpinzon

- 投稿日時: 2026-05-22 12:02:07.270000
- 投票数: 0
- コメントID: `3462181`

<p>and this is how it looks in the gui: submission.csv(338.43 kB) 
<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F636051%2F56604b85744a03da9bc238ede8d16b63%2FScreenshot%202026-05-22%20at%207.01.33AM.png?generation=1779451324362593&alt=media" alt=""></p>
<p>I am using polars instead of pandas to generate the csv</p>

### コメント 2 — Luis Diambra

- 投稿日時: 2026-05-22 15:30:20.300000
- 投票数: -1
- コメントID: `3462249`

<p>I have the same problem; I got "Submission Scoring Error" with the correct file format in all attempts in the last three days. Very frustrated with the Kaggle competition</p>
