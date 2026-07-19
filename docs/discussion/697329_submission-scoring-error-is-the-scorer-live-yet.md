# Submission Scoring Error — Is the scorer live yet?

- 投稿者: Abdessamed Zetroni
- 投稿日時: 2026-05-05 19:23:46.344000
- 投票数: 10
- コメント数: 31（取得数: 31）
- トピックID: `697329`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697329](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697329)

## 本文

<p>Hi, I've been trying to submit but keep getting a "Submission Scoring Error". My submission file matches the <code>sample_submission.csv</code> exactly 14,151 rows, correct <code>id</code> and <code>tvt</code> columns, no <code>NaN</code> values, all IDs verified against the sample.</p>
<p>Since this competition just launched and the leaderboard is empty, I'm wondering if the scorer isn't active yet on Kaggle's side. Can the organizers confirm whether submissions are open and the evaluation script is live?</p>
<p>Thanks</p>

## コメント

### コメント 1 — Ryan Holbrook

- 投稿日時: 2026-05-05 22:12:50.100000
- 投票数: 3
- コメントID: `3453771`

<p>Apologies for the scoring problems. The issue should be resolved now. Please let us know if you continue to have trouble or if anything else comes up.</p>

#### コメント 1.1 — Santiago Maniches

- 投稿日時: 2026-05-05 22:27:27.450000
- 投票数: 0
- コメントID: `3453776`

<p>Thank you! Working on it :) </p>

#### コメント 1.2 — Unknown

- 投稿日時: 2026-05-07 17:45:18.867000
- 投票数: 0
- コメントID: `3454689`

_本文なし_

#### コメント 1.3 — Gideon Adom Boateng

- 投稿日時: 2026-05-11 08:49:45.470000
- 投票数: 0
- コメントID: `3456110`

<p>Still facing scoring issues. My notebook successfully rans and generates the submission.csv but the scorer always fails this is my fifth time and the issue still persists. Kindly help out.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F33604740%2F97dab98b7b48550b07b0d42ed1bc936e%2FScreenshot%20from%202026-05-11%2008-46-11.png?generation=1778489331623087&alt=media" alt=""></p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F33604740%2Fc51e6781b17ff185da77711c81ff9b79%2FScreenshot%20from%202026-05-11%2008-46-01.png?generation=1778489357162169&alt=media" alt=""></p>

### コメント 2 — parijit

- 投稿日時: 2026-05-07 11:26:32.837000
- 投票数: 1
- コメントID: `3454501`

<p>Hi
I dont see any errors when submitting the file however i have problems during the scoring process. The scoring process is still running and it has been a bit over 4 hours, that the scoring has not been completed. 
The kaggle notebook was completed in 6 hours.</p>

#### コメント 2.1 — Gideon Adom Boateng

- 投稿日時: 2026-05-07 12:46:50.050000
- 投票数: 0
- コメントID: `3454547`

<p>Facing similar issue here</p>

### コメント 3 — Santiago Maniches

- 投稿日時: 2026-05-05 21:31:32.903000
- 投票数: 1
- コメントID: `3453744`

<p>Same situation here. Can you please check? Thank you!</p>

### コメント 4 — inversion

- 投稿日時: 2026-05-05 20:22:32.520000
- 投票数: 1
- コメントID: `3453707`

<p>Investigating . . .</p>

### コメント 5 — takaito

- 投稿日時: 2026-05-05 20:00:14.760000
- 投票数: 1
- コメントID: `3453689`

<p>I also failed to submit it.</p>

### コメント 6 — Chris Deotte

- 投稿日時: 2026-05-05 19:56:00.617000
- 投票数: 1
- コメントID: `3453688`

<p>I agree something seems wrong. I have tried multiple times to get my XGB starter to submit but i keep getting scoring errors. </p>
<p>Has anyone tried to just submit the sample submission? If that fails, there is certainly a problem.</p>
<pre><code>pd.read_csv("sample_submission.csv").to_csv("submission.csv",index=False)
</code></pre>
<p>UPDATE: Fixed</p>

#### コメント 6.1 — Chris Deotte

- 投稿日時: 2026-05-05 20:06:45.197000
- 投票数: 3
- コメントID: `3453693`

<p>I tried submitting the sample submission, it failed <a href="https://www.kaggle.com/code/cdeotte/submit-sample-submission">here</a></p>
<p>Admins, can you fix the LB? Thanks <a href="https://www.kaggle.com/addisonhoward">@addisonhoward</a> <a href="https://www.kaggle.com/inversion">@inversion</a> <a href="https://www.kaggle.com/sohiermse">@sohiermse</a></p>
<p>UPDATE: Fixed</p>

##### コメント 6.1.1 — Abdessamed Zetroni

- 投稿日時: 2026-05-05 20:08:58.640000
- 投票数: 1
- コメントID: `3453695`

<p>Thanks for the validation Chris</p>

#### コメント 6.2 — Unknown

- 投稿日時: 2026-06-29 20:50:20.060000
- 投票数: 0
- コメントID: `3483993`

_本文なし_

### コメント 7 — Pavlo Ivanin

- 投稿日時: 2026-05-05 21:52:35.293000
- 投票数: 2
- コメントID: `3453757`

<p>Yes, I also failed to submit</p>

### コメント 8 — Sean

- 投稿日時: 2026-05-31 02:07:42.907000
- 投票数: 0
- コメントID: `3464705`

<p>I am also facing the Submission Scoring Error as of May 30. The column names, row numbers and so on are exactly the same as the sample submission file though. Is there someone who hits the same problem?</p>

### コメント 9 — POR160893

- 投稿日時: 2026-05-08 12:11:08.410000
- 投票数: 0
- コメントID: `3455013`

<p>Hello again,</p>
<p>I wanted to provide a more detailed update because I am STILL consistently getting “Submission Scoring Error” despite extensive debugging and validation.</p>
<p>At this point I do not believe the issue is caused by my actual model predictions.</p>
<p>I reviewed this entire thread carefully and noticed several important things:</p>
<p>• Multiple users reported the exact same issue<br>
• Chris Deotte confirmed even the untouched sample submission failed initially<br>
• Kaggle staff acknowledged scorer problems existed<br>
• Other users are still reporting scoring-related issues and stuck submissions</p>
<p>Because of this, I performed a very extensive set of validation tests on my side.</p>
<p>Here is everything I have verified:</p>
<p>SUBMISSION STRUCTURE CHECKS
• Exactly 14,151 rows
• Exactly 2 columns
• Columns are:</p>
<ul>
<li>id</li>
<li>tvt
• Column order matches sample_submission.csv exactly
• ID order matches sample_submission.csv exactly
• No duplicate IDs
• No missing IDs
• No NaN values
• tvt column is numeric float64
• submission.csv successfully written to:
/kaggle/working/submission.csv</li>
</ul>
<p>I also wrote validation scripts to compare my submission directly against sample_submission.csv including:
• shape checks
• dtype checks
• missing value checks
• duplicate checks
• ID set checks
• ID order checks
• numeric validity checks
• infinity/NaN checks</p>
<p>Everything passes.</p>
<p>Originally I suspected my actual model output (Phase 6 Delta Method) might be causing the issue because it produced only a few unique prediction values.</p>
<p>However, after further debugging I discovered that EVEN COMPLETELY TRIVIAL submissions still fail.</p>
<p>I tested all of the following:</p>
<ol>
<li>Original Phase 6 model submission</li>
<li>Simplified/cleaned Phase 6 submission</li>
<li>Submission with constant numeric values only</li>
<li>Submission generated directly from sample_submission.csv</li>
<li>Completely brand-new minimal notebook with only a few lines of code</li>
</ol>
<p>Example of the minimal notebook I tested:</p>
<p>import pandas as pd</p>
<p>sample = pd.read_csv(
    "/kaggle/input/competitions/rogii-wellbore-geology-prediction/sample_submission.csv"
)</p>
<p>sub = sample.copy()
sub["tvt"] = 11750.0</p>
<p>sub.to_csv("/kaggle/working/submission.csv", index=False)</p>
<p>print(sub.shape)
print(sub.isnull().sum())</p>
<p>This STILL results in:
“Submission Scoring Error”</p>
<p>I also confirmed:
• notebook execution completes
• submission.csv exists
• file is readable
• row count is correct
• IDs match perfectly
• no nulls exist</p>
<p>At this stage, it seems unlikely that my actual model predictions are the root cause.</p>
<p>Could Kaggle staff or the competition hosts please confirm:</p>
<ol>
<li>Is the scorer definitely fully operational right now?</li>
<li>Have any recent submissions successfully scored?</li>
<li>Are there any hidden validation rules beyond the documented CSV format?</li>
<li>Is there any known intermittent issue with the leaderboard/scoring backend?</li>
</ol>
<p>I’ve now used 11+ submission attempts debugging this and wanted to provide as much technical detail as possible.</p>
<p>Thanks very much for any clarification.</p>

### コメント 10 — Navneet

- 投稿日時: 2026-05-08 07:52:49.770000
- 投票数: 0
- コメントID: `3454925`

<p>Thanks for the information on Submission Scoring Error <a href="https://www.kaggle.com/abdessamedzetroni">@abdessamedzetroni</a> </p>

### コメント 11 — Gideon Adom Boateng

- 投稿日時: 2026-05-07 12:45:59.430000
- 投票数: 0
- コメントID: `3454546`

<p>Same issue here. Code was able to generate the submission.csv but I could not get a score because it failed. Kindly help out</p>

### コメント 12 — POR160893

- 投稿日時: 2026-05-07 11:22:39.343000
- 投票数: 0
- コメントID: `3454499`

<p>Hi everyone,</p>
<p>I’m still experiencing persistent “Submission Scoring Error” issues in this competition after making 11 submissions up to this stage and wanted to provide a detailed summary of everything I have tested so far.</p>
<p>At this point, I no longer think the issue is caused by my actual model predictions, because even extremely simplified submissions are failing.</p>
<p>Here is everything I have checked:</p>
<p>• Verified the exact expected submission format using sample_submission.csv
• Confirmed my submission has exactly 14,151 rows and 2 columns
• Confirmed columns are exactly:</p>
<ul>
<li>id</li>
<li>tvt</li>
</ul>
<p>• Confirmed column order matches sample_submission.csv exactly
• Confirmed ID order matches sample_submission.csv exactly
• Confirmed all IDs are present with no duplicates
• Confirmed there are no missing values or NaNs
• Confirmed tvt is numeric float64
• Confirmed submission.csv is written to:
/kaggle/working/submission.csv</p>
<p>I also built validation code to compare my submission against sample_submission.csv directly, including:</p>
<ul>
<li>shape checks</li>
<li>dtype checks</li>
<li>duplicate checks</li>
<li>missing value checks</li>
<li>ID set/order checks</li>
<li>numeric validity checks</li>
</ul>
<p>Everything passes.</p>
<p>I originally suspected my Phase 6 Delta Method model predictions might be the issue because the predictions only had a few unique values. However, after further testing, I discovered the same scoring error occurs even when I completely bypass the model.</p>
<p>I tested all of the following:</p>
<ol>
<li>Original Phase 6 submission</li>
<li>Cleaned Phase 6 submission with simplified float formatting</li>
<li>Submission using constant values only</li>
<li>Submission created directly from sample_submission.csv</li>
<li>Brand new minimal notebook with only a few lines of code</li>
</ol>
<p>For example, I tested this extremely minimal notebook:</p>
<p>import pandas as pd</p>
<p>sample = pd.read_csv(
"/kaggle/input/competitions/rogii-wellbore-geology-prediction/sample_submission.csv"
)</p>
<p>sub = sample.copy()
sub["tvt"] = 11750.0</p>
<p>sub.to_csv("/kaggle/working/submission.csv", index=False)</p>
<p>This still produces:
“Submission Scoring Error”</p>
<p>I also confirmed:</p>
<ul>
<li>the notebook completes successfully</li>
<li>submission.csv exists</li>
<li>the file is readable</li>
<li>row counts are correct</li>
<li>there are no nulls</li>
</ul>
<p>Additionally, I noticed older discussion threads where multiple users reported the same issue, including failures when submitting the untouched sample_submission.csv itself.</p>
<p>Because of this, I’m wondering if:</p>
<ul>
<li>the scorer may still have intermittent issues,</li>
<li>hidden validation rules may exist beyond the documented format,</li>
<li>or there is some backend evaluation problem still occurring.</li>
</ul>
<p>Can Kaggle staff or competition hosts confirm:</p>
<ol>
<li>that the scorer is fully operational now,</li>
<li>whether additional hidden constraints exist for tvt values,</li>
<li>and whether anyone has successfully submitted a minimal constant-value submission recently?</li>
</ol>
<p>Thanks — I wanted to provide as much debugging detail as possible before continuing to rewrite my pipeline.</p>

#### コメント 12.1 — Ryan Holbrook

- 投稿日時: 2026-05-08 11:47:38.907000
- 投票数: 0
- コメントID: `3455004`

<p>Hi <a href="https://www.kaggle.com/por160893">@por160893</a>,</p>
<p>The scoring issue should be fully resolved now; it was only present for a few hours after launch. Have you tried your minimal submission that just submits the <code>sample_submission.csv</code> lately? I imagine it should work. Another option is to copy one of the community notebooks (like <a href="https://www.kaggle.com/code/cdeotte/xgb-starter-cv-15">this one</a>, say) and try submitting it.</p>

##### コメント 12.1.1 — POR160893

- 投稿日時: 2026-05-08 13:07:47.580000
- 投票数: 0
- コメントID: `3455036`

<p>Hi <a href="https://www.kaggle.com/RyanHolbrook">@RyanHolbrook</a> and organizers,</p>
<p>I am still encountering a submission scoring error even after rebuilding the notebook from scratch and validating the submission file extensively.</p>
<p>What I tested:</p>
<ul>
<li>I deleted all previous notebook cells and created a completely clean notebook.</li>
<li>I successfully submitted the sample submission earlier using the same notebook workflow.</li>
<li>I generated my actual predictions in RStudio locally and uploaded the CSV to Kaggle as a dataset.</li>
<li>In the Kaggle Python notebook, I simply load the uploaded CSV and write it directly to /kaggle/working/submission.csv.</li>
</ul>
<p>Validation checks completed successfully:</p>
<ul>
<li>Shape matches sample exactly: (14151, 2)</li>
<li>Columns exactly match: id, tvt</li>
<li>No missing values</li>
<li>No duplicate IDs</li>
<li>ID order matches sample submission</li>
<li>id dtype = object</li>
<li>tvt dtype = float64</li>
<li>Output file exists successfully before submission</li>
</ul>
<p>Notebook output confirms:</p>
<ul>
<li>submission.csv is written correctly</li>
<li>Kaggle notebook execution completes normally</li>
<li>No runtime errors occur</li>
</ul>
<p>Example predictions:
text 000d7d20_1442    11749.007289 000d7d20_1443    11749.007289 000d7d20_1444    11749.007289 </p>
<p>One thing I noticed:</p>
<ul>
<li>My Phase 6B model currently produces only 4 unique prediction values across all 14,151 rows.</li>
<li>However, the values are numeric and valid floats.</li>
</ul>
<p>The confusing part is:</p>
<ul>
<li>the sample submission workflow succeeds,</li>
<li>but this actual prediction file still produces a scoring error.</li>
</ul>
<p>Could the scorer be rejecting submissions with low prediction variance or certain value ranges?</p>
<p>Any guidance would be greatly appreciated, because structurally the submission appears fully valid now.</p>

##### コメント 12.1.2 — Ryan Holbrook

- 投稿日時: 2026-05-08 14:47:10.790000
- 投票数: 1
- コメントID: `3455071`

<p>Hi <a href="https://www.kaggle.com/por160893">@por160893</a>,</p>
<blockquote>
  <p>I generated my actual predictions in RStudio locally and uploaded the CSV to Kaggle as a dataset.</p>
</blockquote>
<p>In this competition, you need to submit a notebook that creates your submission file. The test set you see on the Data page isn't the actual test set; those are just some examples drawn from the training set. When your submitted notebook is scored, this example data is replaced with the actual hidden test set. Your submissions are failing because you're making predictions on the wrong test data; the prediction ids don't match and there aren't enough of them.</p>

### コメント 13 — Nikita Shevyrev

- 投稿日時: 2026-05-07 05:18:24.297000
- 投票数: 0
- コメントID: `3454420`

<p>Hi everyone,</p>
<p>I’m trying to understand the submission behavior for this code competition.</p>
<p>Yesterday I had 5 failed submission attempts. Today I launched a new submission and noticed what looks like two separate processes:</p>
<ol>
<li>one process running my notebook computation;</li>
<li>another one already showing the competition submission as “Scoring…”</li>
</ol>
<p>The confusing part is that the notebook is still running and has not produced <code>submission.csv</code> yet, but it already looks like one submission attempt has been used.</p>
<p>Could someone clarify whether this is expected behavior? Does Kaggle start the scoring/submission process in parallel with the notebook rerun, and does this still count as only one submission attempt?</p>
<p><a href="https://www.kaggle.com/ryanholbrook">@ryanholbrook</a> Sorry to bother you directly, but I wanted to check whether I may have done something wrong on my side or if this is normal for code competitions.</p>
<p>Thank you!</p>

#### コメント 13.1 — Ryan Holbrook

- 投稿日時: 2026-05-08 11:50:38.467000
- 投票数: 0
- コメントID: `3455007`

<p>There are two runs kicked off when you submit from the notebook editor. One of them is a run on the published data that you see on the Data page. The other is the rerun on the hidden test data; that's the run showing as "Scoring…". These two sessions run in parallel, but only the run on the hidden test set counts against your submission attempts.</p>
<p>It's also possible to do these separately. You would just commit the notebook normally, and after it's finished, go to the notebook viewer and submit from there. That would kick off the rerun on the hidden test set only.</p>

### コメント 14 — Chattso-GPT (Yasuhito Yanagisawa)

- 投稿日時: 2026-05-07 01:16:40.407000
- 投票数: 0
- コメントID: `3454362`

<p>I couldn’t even submit because of an error lol</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F9304974%2F5fb6fae17f104dafb3333942e1f6b133%2Ferror.png?generation=1778116536533794&alt=media" alt=""></p>

#### コメント 14.1 — Chattso-GPT (Yasuhito Yanagisawa)

- 投稿日時: 2026-05-07 16:56:54.843000
- 投票数: 0
- コメントID: `3454670`

<p>I’ve been stuck with this error all day and can’t get anything done… even in other competitions.</p>

##### コメント 14.1.1 — Ryan Holbrook

- 投稿日時: 2026-05-07 17:10:20.680000
- 投票数: 0
- コメントID: `3454675`

<p>Where are you seeing this error and what were you doing at the time?</p>

##### コメント 14.1.2 — Chattso-GPT (Yasuhito Yanagisawa)

- 投稿日時: 2026-05-07 17:52:03.403000
- 投票数: 0
- コメントID: `3454690`

<p>I always get this error whenever I try to create or operate a notebook. It’s been happening constantly since yesterday.</p>

##### コメント 14.1.3 — Ryan Holbrook

- 投稿日時: 2026-05-08 17:04:45.627000
- 投票数: 0
- コメントID: `3455139`

<p>I'm almost wondering if this is an authentication issue or something to do with a stale cache. Maybe try logging out and logging in again, or try logging in under an incognito session and see if that helps.</p>

##### コメント 14.1.4 — Chattso-GPT (Yasuhito Yanagisawa)

- 投稿日時: 2026-05-09 17:07:39.793000
- 投票数: 0
- コメントID: `3455558`

<p>Re-logging in didn’t help, but using an incognito session fixed it. Thank you very much!</p>

### コメント 15 — Bryce Chambers

- 投稿日時: 2026-05-07 00:45:07.830000
- 投票数: 0
- コメントID: `3454355`

<p>Can we upload just a .csv file?  Or is command line only required?</p>
