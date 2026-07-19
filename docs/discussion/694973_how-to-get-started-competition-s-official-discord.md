# How to get started + Competition's Official Discord

- 投稿者: María Cruz
- 投稿日時: 2026-04-27 21:33:19.370000
- 投票数: 15
- コメント数: 21（取得数: 21）
- トピックID: `694973`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/694973](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/694973)

## 本文

<h3>Information for newbies</h3>
<p><strong>New to machine learning and data science?</strong> No question is too basic or too simple. Feel free to start your own thread, or use this thread as a place to post any first-timer clarifying questions for the Kaggle community to help you with!</p>
<p><strong>New to Kaggle?</strong> Take a look at a few videos to learn a bit more about <a href="https://www.youtube.com/watch?v=aIus8si_Et0">site etiquette</a>, <a href="https://www.youtube.com/watch?v=sEJHyuWKd-s">Kaggle lingo</a>, and <a href="https://www.youtube.com/watch?&v=GJBOMWpLpTQ">how to enter a competition using Kaggle Notebooks</a>. Publish and share your <a href="https://www.kaggle.com/docs/models#publishing-a-model">models on Kaggle Models</a>!</p>
<p><strong>Looking for a team?</strong> Express your interest in joining a team through our <a href="https://www.kaggle.com/discussions/product-feedback/341195">Team Up</a> feature.</p>
<p><strong>Remember</strong>: Kaggle is for everyone. Whether you're teaming up or sharing tips in the competition forum, we expect everyone to follow our Kaggle community guidelines.</p>
<h3>Competition's Official Discord</h3>
<p>In addition to this competition forum, you can continue the discussion in our official Kaggle Discord Server here:</p>
<h1><a href="http://discord.gg/kaggle">discord.gg/kaggle</a></h1>
<p>The Discord is a great place to ask getting started questions, chat about the nuances of this competition, and connect with potential team mates. Learn more about Discord at our <a href="https://www.kaggle.com/discussions/general/429933">announcement here</a>. Here are a few things to keep in mind though:</p>
<p><strong>1. Discord Competition Channels are 'Public' - Don't Share Private Information</strong></p>
<p>Discord channels for specific competitions are considered 'public' spaces where you are allowed to talk about competition details. Please remember that private sharing of competition code or data outside of your team is, as always, not permitted. Code sharing must always be done publicly through the Kaggle forums/notebooks.</p>
<p><strong>2. Discord Competition Channels are Not Monitored by Staff - Keep Important Information on the Kaggle Forums</strong></p>
<p>Kaggle Staff and Hosts running competitions will not monitor Discord or be available to answer questions in Discord. This is intended to be a more casual space to discuss competitions and help each other. Please keep important questions, insights, writeups, and other valuable conversation on the Kaggle forums. </p>
<p>Happy modeling! </p>

## コメント

### コメント 1 — Luis Diambra

- 投稿日時: 2026-05-20 19:54:01.800000
- 投票数: 1
- コメントID: `3461549`

<p>Hi Team. This is my first time participating in Kaggle. I've submitted file, but but I'm getting a "submission score error" message, even though the file has the correct format. When I check the log file, I don't see any reports of what's wrong. How can I find out what's wrong with the scoring?</p>

#### コメント 1.1 — PC Jimmmy

- 投稿日時: 2026-05-20 23:27:07.833000
- 投票数: 1
- コメントID: `3461604`

<p>The log files of no value for a submission error, assuming the the save and run worked ok.</p>
<p>The best way to get help is to make your notebook public and than add a link here in the discussion post.  It's pretty hard to trouble shoot without it.</p>

##### コメント 1.1.1 — Luis Diambra

- 投稿日時: 2026-05-21 13:14:28.377000
- 投票数: 0
- コメントID: `3461758`

<p>Thank you very much Jimmy, I made public my notebook:
<a href="https://www.kaggle.com/code/luisdiambra/notebookb7f2cbd34e">https://www.kaggle.com/code/luisdiambra/notebookb7f2cbd34e</a>
I am becoming crazy; now the systems says "This Competition requires a submission file named submission.csv and the selected Notebook Version does not output this file. …" I can see the correct file generated! In other opportunity the problem was a scoring error. I can not understand this environmemt. Please can you see what I am doing wrong in my code?</p>

##### コメント 1.1.2 — PC Jimmmy

- 投稿日時: 2026-05-21 14:12:09.900000
- 投票数: 1
- コメントID: `3461790`

<p>playing with it now - will hollar if I figure out anything </p>

##### コメント 1.1.3 — PC Jimmmy

- 投稿日時: 2026-05-21 14:35:27.470000
- 投票数: 0
- コメントID: `3461800`

<p>I agree the file exists.  But you need to remember that the submission.csv file you see is generated using only a couple of wells.  The real submission works on a larger count of files.  Pretty common that you have a shape error somewhere that results in no submission file being created
.  Still looking……</p>

##### コメント 1.1.4 — PC Jimmmy

- 投稿日時: 2026-05-21 14:37:57.297000
- 投票数: 0
- コメントID: `3461801`

<p>Found this issue - trying to revise the code - the error output when we do a submission is always pretty NOT useful :)  Past practice from years ago - skilled folks could use error messages to "cheat".</p>
<p>If a hidden test well fails for any reason during the loop in Cell 2, its _predicted.csv file is never written. When Cell 3 tries to merge everything, it hits a FileNotFoundError and the entire notebook crashes, resulting in no submission.csv being written.</p>

##### コメント 1.1.5 — PC Jimmmy

- 投稿日時: 2026-05-21 14:41:40.440000
- 投票数: 0
- コメントID: `3461803`

<p>The first submission I made with no changes - this is the error output that you should share when seeking help rather than just submission score error.</p>
<p>*Your notebook generated a submission file with incorrect format. Some examples causing this are: wrong number of rows or columns, empty values, an incorrect data type for a value, or invalid submission values from what is expected. *</p>

##### コメント 1.1.6 — PC Jimmmy

- 投稿日時: 2026-05-21 14:50:57.787000
- 投票数: 0
- コメントID: `3461811`

<p>My attempt to fix the first issue did not work - I got the exact same full error message.</p>
<p>I have run out of submissions for the day - will look again later when a new kaggle day starts.  Found a second issue before I did the submit.</p>
<p>When you sort the DataFrame df_h by 'MD', the indices are shuffled. Calling idxmax() returns the label of the index, not the integer position. If you use start_idx - 5 to slice, pandas can misinterpret this as a positional slice on a shuffled index, pulling the wrong rows entirely (or negative slicing, which grabs from the bottom of the data frame instead of the top.</p>
<p>This is the <a href="https://www.kaggle.com/code/pcjimmmy/notebookb7f2cbd34e">changed notebook</a> I tried in cell 2 that fixed two issues I ended up finding - but they are not the killers of the submission.csv :)  They might have just resulted in poor predictions</p>

##### コメント 1.1.7 — PC Jimmmy

- 投稿日時: 2026-05-21 14:57:53.677000
- 投票数: 1
- コメントID: `3461812`

<p>I made a mistake in cell 3 attempting to fix the index issue.  The link above should take you to a "better fix" :)  Will do the submission in around 12 hours.</p>

##### コメント 1.1.8 — Unknown

- 投稿日時: 2026-05-21 15:48:49.390000
- 投票数: 0
- コメントID: `3461834`

_本文なし_

##### コメント 1.1.9 — Luis Diambra

- 投稿日時: 2026-05-21 16:01:09.203000
- 投票数: 0
- コメントID: `3461838`

<p>This comment is valuable; I've never seen it before. I don't know how you got it or how to access it. Even so, it wasn't helpful in determining the problem because the submission files I sent have 2 columns and 14,151 rows (not counting the header row). There are no missing values ​​either. It should be OK.</p>

##### コメント 1.1.10 — Luis Diambra

- 投稿日時: 2026-05-21 16:03:48.140000
- 投票数: 0
- コメントID: `3461840`

<p>Many thanks for the comments. That is my first doubt. There are 3 wells in the test set. Should the submission file only contain the prediction for these 3 wells, as indicated in the sample_submission.csv file? Or are there more wells to predict ?</p>

##### コメント 1.1.11 — PC Jimmmy

- 投稿日時: 2026-05-22 00:32:03.553000
- 投票数: 0
- コメントID: `3462025`

<p>If you click on the words Submission Scoring Error just below the notebook name on the submissions tab it would take you to the additional comments.</p>
<p>What you can only see are the 3 wells in the test set that kaggle provides to catch some of the errors that might be made.  For the actual scoring a folder with larger number of test wells is used that we are never able to view.</p>

##### コメント 1.1.12 — PC Jimmmy

- 投稿日時: 2026-05-22 00:39:20.673000
- 投票数: 1
- コメントID: `3462027`

<p>My attempt has same error.  Don't think I can figure it out.  Sorry</p>

#### コメント 1.2 — José Luiz Luna-Xavier

- 投稿日時: 2026-05-21 16:51:13.620000
- 投票数: 0
- コメントID: `3461861`

<p>Hi Luis, I checked your notebook and logs. The notebook seems to run until the end: the logs do not show a fatal Python crash, only pandas warnings. The issue is most likely with the generated submission.csv file.</p>
<p>Please make sure that submission.csv has exactly the required columns: id,tvt, the same number of rows as sample_submission.csv, the same ID order, no duplicated IDs, no missing NaN values, and no infinite values.</p>
<p>In your code, some missing GR gaps may leave NaN values in TVT_predicted, which can then propagate into the final tvt column. I suggest adding a strict validation cell right after creating submission.csv, and filling/interpolating any missing predictions before exporting the final file.</p>
<p>The priority is not improving the model yet; first make sure the submission file is 100% valid for Kaggle.</p>

##### コメント 1.2.1 — Luis Diambra

- 投稿日時: 2026-05-21 17:54:22.257000
- 投票数: 0
- コメントID: `3461887`

<p>Thank you so much for looking at my code. I've already addressed the NAN in GR, filling in the gaps. I've added a quality control check to the end of my notebook for my submission.csv file, checking for various potential issues (NAN, inf, incomplete, etc.), but I still haven't found the problem with the submission file.</p>

##### コメント 1.2.2 — José Luiz Luna-Xavier

- 投稿日時: 2026-05-21 18:35:26.737000
- 投票数: 0
- コメントID: `3461910`

<p>I checked the CSV file you shared. The file itself looks structurally valid: it has exactly the columns id,tvt, 14,151 rows, 14,151 unique IDs, no duplicates, no NaN values, no infinite values, and the three expected test wells with continuous row indices.</p>
<p>So the issue is probably not the CSV content. Please make sure the notebook saves the file exactly as /kaggle/working/submission.csv, then run Save Version and confirm that submission.csv appears in the notebook outputs.</p>
<p>Some TVT predictions have large jumps, especially in well 000d7d20, which may affect RMSE, but this should not cause a Kaggle submission-format error. Go to "Submit to competition" section and good luck, Luis!</p>

##### コメント 1.2.3 — Luis Diambra

- 投稿日時: 2026-05-21 19:12:51.697000
- 投票数: 0
- コメントID: `3461926`

<p>Thanks for your efforts in helping me. Unfortunately, it still gives the same problem. My output is generated in the /kaggle/working/ subdir as submission.csv. I tested again with this new public notebook: 
<a href="https://www.kaggle.com/code/luisdiambra/notebook02efe0e09d">https://www.kaggle.com/code/luisdiambra/notebook02efe0e09d</a>
It is a pity, but this is the second day with the same problem. I have no more submissions for today (5/5), very frustrating.</p>

### コメント 2 — Tiago Soares

- 投稿日時: 2026-07-02 19:28:32.690000
- 投票数: 0
- コメントID: `3486670`

<p>Hi <a href="https://www.kaggle.com/macruzbar">@macruzbar</a>, once the competition ends is it allowed for us to publish our solution for this particular competition on a public repository such as github to show to potential employers?</p>

#### コメント 2.1 — María Cruz

- 投稿日時: 2026-07-10 20:35:28.113000
- 投票数: 0
- コメントID: `3494858`

<p>Hi <a href="https://www.kaggle.com/tiagosoares">@tiagosoares</a> -- after the competition ends, if you have a winning solution, you have to observe the rules. Specifically the <a href="https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/rules#5.-winner-license">winner license</a>. If yours is not a winning solution, the code is yours to share as you please. Let me know if you have any other questions!</p>
<p>Good luck, </p>
<p>María</p>

##### コメント 2.1.1 — Tiago Soares

- 投稿日時: 2026-07-10 23:54:51.410000
- 投票数: 0
- コメントID: `3494894`

<p>Thanks Maria, I guess it will be clearer once the competition ends, my score will definitely be much higher than what it is at the moment. Cheers</p>
