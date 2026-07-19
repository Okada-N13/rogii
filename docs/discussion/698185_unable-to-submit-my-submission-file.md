# Unable to submit my submission file

- 投稿者: Chesang Irine
- 投稿日時: 2026-05-08 19:37:33.793000
- 投票数: 2
- コメント数: 7（取得数: 7）
- トピックID: `698185`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698185](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698185)

## 本文

<p>I have downloaded my CSV file and am ready to submit it, but I’ve noticed that the upload button on the prediction submission page is not active. What should I do in this case? Also, whenever I try the quick option of “Save & Commit,” the submission keeps failing.</p>

## コメント

### コメント 1 — Aly Ayman

- 投稿日時: 2026-06-08 20:15:02.013000
- 投票数: -2
- コメントID: `3468338`

<p>Direct CSV upload is disabled for this competition by design. Submissions must come from a committed notebook that produces a file named exactly <code>submission.csv</code>. So instead of uploading, you'll submit from a notebook.</p>
<p>A quick tip: if a full commit keeps failing and you're not sure why, open the failed version and read the log — it usually points to the exact cell that errored.
If you share the error message from the commit log, I'm happy to help narrow it down further. Good luck!</p>

### コメント 2 — Olena Arshynnikova

- 投稿日時: 2026-06-09 23:35:18.043000
- 投票数: 0
- コメントID: `3468850`

<p>I can not submit my submission file (submit button is inactive). My notebook is turned off from the internet, it has a submission file (csv format). When I re-run all cells of my notebook, everything ok. Please explain to me how you could upload your submission file</p>

#### コメント 2.1 — PC Jimmmy

- 投稿日時: 2026-06-10 01:16:21.897000
- 投票数: 0
- コメントID: `3468860`

<ol>
<li>Save Version - close the notebook.   Depending on your code it should be done in a few minutes as it's only predicting on the 3 fake wells.</li>
<li>Go back to the notebook - Output tab - IF you only have saved a single file than the submission.csv would be selected and the submit button should be alive.  If you happen to have saved any files besides the submission than you would need to click on the submission.csv to activate the button.</li>
</ol>

### コメント 3 — PatrickAIForFun

- 投稿日時: 2026-05-09 14:39:54.433000
- 投票数: 0
- コメントID: `3455483`

<p>This is a code competition. Thus you must submit the notebook itself and not the csv.
This notebook is then re-run automatically with many more and different files in the test folder -> this ensures the test set stays hidden.</p>

### コメント 4 — PC Jimmmy

- 投稿日時: 2026-05-09 00:01:40.583000
- 投票数: 0
- コメントID: `3455262`

<p>The words "downloaded csv file" - you must generate the csv file in a kaggle notebook.   Where did you create the file?</p>

#### コメント 4.1 — Chesang Irine

- 投稿日時: 2026-05-09 12:52:01.540000
- 投票数: 0
- コメントID: `3455457`

<p>from my kaggle notebook…tried saving it many times  /saving the version but kept failing so i decided to download from the kaggle output files with intentions of uploading it manually only to realize the upload button is inactive</p>

##### コメント 4.1.1 — PC Jimmmy

- 投稿日時: 2026-05-09 15:28:46.763000
- 投票数: 0
- コメントID: `3455516`

<p>You cannot upload it manually for this type of competition.<br>
When you run your kaggle notebook does the submission.csv file that was created look similar to what has been shared on all the code notebooks?   </p>
<p>After I run my notebook I go to the output and select the submission.csv file (I create some extra files so kaggle would not be certain which is my submission).   Once I have selected the file and it looks ok - than I hit the Submit to Completion button.</p>
