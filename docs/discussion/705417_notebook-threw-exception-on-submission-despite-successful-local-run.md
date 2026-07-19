# Notebook Threw Exception on submission despite successful local run

- 投稿者: Alexander Osorio
- 投稿日時: 2026-06-10 00:37:20.573000
- 投票数: 2
- コメント数: 4（取得数: 4）
- トピックID: `705417`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/705417](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/705417)

## 本文

<p>My notebook runs successfully (729s, generates submission.csv with 14151 rows) but every submission attempt shows "Notebook Threw Exception". The notebook logs show green checkmarks and correct output. This has happened across 5+ versions. Is there a known issue with submissions for this competition?</p>

## コメント

### コメント 1 — OpPrime

- 投稿日時: 2026-06-10 07:45:38.033000
- 投票数: 1
- コメントID: `3468918`

<p>I would run it cell by cell, also make sure you have not muted warnings so you can see what pops up.</p>
<p>Then also look at if you are using P100s, and if you are getting a torch sm_60 accelerator error (read and run <a href="https://www.kaggle.com/code/mauriciooffermann/kaggle-cuda-compat-probe-p100-torch">this</a> to understand the issue if that is your problem). </p>

### コメント 2 — PC Jimmmy

- 投稿日時: 2026-06-10 16:04:43.847000
- 投票数: 0
- コメントID: `3469086`

<p>If you cannot figure it out - best advice-  make the notebook public and put a link in this discussion.  It's very hard to troubleshoot code you cannot see :)</p>
<p>Had there been a link it's likely someone as good a Chris could have taken a look and spotted the issue.</p>

### コメント 3 — PC Jimmmy

- 投稿日時: 2026-06-10 01:08:51.040000
- 投票数: 0
- コメントID: `3468859`

<p>More likely issue is that your getting a memory error or shape error when the real test data is used.   The 3 fake test wells supplied might not be enough to stress your code.  Once the full (200 maybe) set of test wells evaluated there's lots of errors that will show up as exceptions.</p>

#### コメント 3.1 — Chris Deotte

- 投稿日時: 2026-06-10 14:56:39.287000
- 投票数: 1
- コメントID: `3469046`

<p>I agree. Probably a memory error. You can find it my using your inference notebook to infer 200 train wells. That will simulate what your inference notebook does during submit.</p>
