# Test well 000d7d20 also exists in train/  (intentional?)

- 投稿者: Maher el Ouahabi
- 投稿日時: 2026-05-06 12:49:35.680000
- 投票数: 3
- コメント数: 5（取得数: 5）
- トピックID: `697507`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697507](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697507)

## 本文

<p>Hi <a href="https://www.kaggle.com/igorkuvaev">@igorkuvaev</a> <a href="https://www.kaggle.com/sergeyalyaev">@sergeyalyaev</a>,</p>
<p>While exploring the data I noticed that the well_id <code>000d7d20</code> appears
both in <code>train/</code> and in <code>test/</code>. The horizontal trajectory (MD, X, Y, Z)
matches row-for-row across the two folders for all 5,278 rows; the GR
log differs in ~43% of rows (likely intentional noise).</p>
<p>Two questions:</p>
<ol>
<li>Is the visible test/000d7d20 a duplicate of train/000d7d20 by design
(e.g., a sanity sample for kernel debugging), or is it an accidental
carry-over that will be re-mounted as a new well at hidden-test
scoring time?</li>
<li>The other two visible test wells (00bbac68, 00e12e8b) do not appear
in train/. Is the hidden test set composed of mostly new wells, or
is some overlap with train expected (consistent with synchronous
reruns + held-out split)?</li>
</ol>
<p>Asking because this directly affects whether the right strategy is to
condition predictions on <code>train/<well_id></code> existence, or to ignore that
path entirely. Want to make sure we're modeling the right problem.</p>
<p>Thanks!</p>

## コメント

### コメント 1 — Ryan Holbrook

- 投稿日時: 2026-05-06 14:00:02.230000
- 投票数: 5
- コメントID: `3454052`

<p>Hi <a href="https://www.kaggle.com/maherelouahabi">@maherelouahabi</a>,</p>
<p>The wells you see in <code>test/</code> are are just example data to help you author your submissions. When you submit your notebook, the example test data will be replaced with the actual test data, which does not contain wells in the training set.</p>

#### コメント 1.1 — MasonLeSaint

- 投稿日時: 2026-06-12 02:56:49.123000
- 投票数: 0
- コメントID: `3471639`

<p>Hi <a href="https://www.kaggle.com/ryanholbrook">@ryanholbrook</a>!
Could you share how many well in the actual test dataset will run? If I know that information I can optimize my scoring code because now I see that scoring time is more than 2 hours but it still doesnt finish. I want to investigate where is problem.
Thank you.</p>

##### コメント 1.1.1 — MasonLeSaint

- 投稿日時: 2026-06-12 03:00:37.497000
- 投票数: 0
- コメントID: `3471641`

<p>I created some function in .py file, upload it and use it to run notebook. So it is acceptable when I run the submission?</p>

##### コメント 1.1.2 — Houngnibo Jaouen

- 投稿日時: 2026-06-18 14:32:11.053000
- 投票数: 0
- コメントID: `3474692`

<p>Environ 200 puits</p>

### コメント 2 — lingyu07

- 投稿日時: 2026-05-06 12:51:29.767000
- 投票数: 0
- コメントID: `3454020`

<p>3 wells :</p>
<p>000d7d20: 3836 hidden rows</p>
<p>00bbac68: 6014 hidden rows</p>
<p>00e12e8b: 4301 hidden rows</p>
