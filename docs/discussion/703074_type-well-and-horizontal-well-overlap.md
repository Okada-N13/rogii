# Type-well and horizontal well Overlap.

- 投稿者: Shrey Gandhi
- 投稿日時: 2026-05-28 13:19:12.158000
- 投票数: 4
- コメント数: 5（取得数: 5）
- トピックID: `703074`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/703074](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/703074)

## 本文

<p>Looking at other discussions and the YT videos, geo-steering seems to be a GR overlap problem from the typewell to the horizontal well. </p>
<p>But on looking at the overlaps of the wells, some wells have terrible overlaps. What do the domain experts use here to decide the TVT values?</p>
<p>Specially the segment between 15500 and 16000.
Example:
well pair_id = '000d7d20'
<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F7200429%2F08c8197360236feb07c13c32cabb075b%2Fterrible_well.png?generation=1779974258651818&alt=media" alt=""></p>

## コメント

### コメント 1 — Chris Deotte

- 投稿日時: 2026-05-28 14:53:51.270000
- 投票数: 4
- コメントID: `3464000`

<p>Well <code>000d7d20</code> has great correlation between GR horizontal and vertical well:
<img src="https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/May-2026/well.png" alt=""></p>

#### コメント 1.1 — Shrey Gandhi

- 投稿日時: 2026-05-28 15:14:25.547000
- 投票数: 1
- コメントID: `3464007`

<p>I see your plot is between TVT and GR. 
Also, is vertical well same as typewell or ar these the values before PS.
Great analysis btw, could be used somewhere.</p>

##### コメント 1.1.1 — Chris Deotte

- 投稿日時: 2026-05-28 15:20:22.300000
- 投票数: 1
- コメントID: `3464011`

<p>What i call "vertical well" is <code>typewell.csv</code>. And what i call "horizontal well" is the TVT before and after PS.</p>
<p>I make this plot with <code>df.groupby('TVT').GR.mean()</code> applied to the two dataframes, <code>horizontal_well.csv</code> and <code>typewell.csv</code></p>

### コメント 2 — Shrey Gandhi

- 投稿日時: 2026-06-24 09:00:47.543000
- 投票数: 0
- コメントID: `3479905`

<p>Any geologist, that would like to comment here?</p>

### コメント 3 — really?

- 投稿日時: 2026-06-12 13:36:51.237000
- 投票数: -1
- コメントID: `3471801`

<p>excellent!</p>
