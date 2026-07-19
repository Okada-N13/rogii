# How to submit notebook for csv file

- 投稿者: MasonLeSaint
- 投稿日時: 2026-06-11 03:23:01.444000
- 投票数: 3
- コメント数: 1（取得数: 1）
- トピックID: `707615`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/707615](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/707615)

## 本文

<p>Hi everyone,
I want to ask about how to submit notebook or submission.csv file. I can not select the notebook and choose submit as a image below <img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F32853119%2Fc2ee43e6a930424670606fb03db59af4%2Fsubmit01.png?generation=1781148176367524&alt=media" alt=""></p>

## コメント

### コメント 1 — yuji y

- 投稿日時: 2026-06-12 07:40:06.120000
- 投票数: 0
- コメントID: `3471711`

<p>This is a <strong>code competition</strong>, so you can't upload a CSV directly ? the submission <em>is</em> a notebook. The flow:</p>
<ol>
<li>Your notebook must be attached to this competition (create it from the competition's Code tab, or add the competition under "Input").</li>
<li>It must write the predictions to <code>/kaggle/working/submission.csv</code> (exact name).</li>
<li><strong>Save Version</strong> (Save & Run All) and wait until the version finishes successfully ? you submit a <em>completed version</em>, not the editor state.</li>
<li>Then either: notebook page Output tab  "Submit to Competition", or Submissions page "Submit Prediction" pick the notebook version.</li>
</ol>
<p>If the notebook can't be selected / the button is greyed out, it's almost always one of:</p>
<ul>
<li>the version run failed or is still running (check the version log),</li>
<li>no <code>submission.csv</code> in the Output of that version,</li>
<li><strong>Internet is ON</strong> in notebook settings (must be OFF for this comp),</li>
<li>the notebook isn't attached to this competition,</li>
<li>you haven't accepted the competition rules.</li>
</ul>
<p>Also note the example <code>test/</code> folder is replaced with the real hidden test set when your notebook is re-run for scoring, so make sure your code discovers wells dynamically (e.g. from <code>sample_submission.csv</code>) instead of hard-coding the 3 example well ids.</p>
