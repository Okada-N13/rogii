# Notebook libraries

- 投稿者: Alchemist
- 投稿日時: 2026-05-29 23:49:20.073000
- 投票数: 2
- コメント数: 1（取得数: 1）
- トピックID: `703335`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/703335](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/703335)

## 本文

<p>Hi,</p>
<p>how can we know the list of libraries and their versions installed in the env in which the submission will run ? 
I've found this link: <a href="https://github.com/Kaggle/docker-python/blob/main/README.md">https://github.com/Kaggle/docker-python/blob/main/README.md</a></p>
<p>but there is no <code>pandas</code> or <code>polars</code> for example, although I see them when I launch a notebook.</p>

## コメント

### コメント 1 — PatrickAIForFun

- 投稿日時: 2026-05-30 07:35:00.147000
- 投票数: 1
- コメントID: `3464533`

<p>The environment during submission is the same as in the kaggle notebook session. Thus, if it works there, it will also work in the submission. If wamt a list of these, just run <code>!pip freeze</code> in a notebook.
If you need additional libraries, there is button/menu where you can add dependencies in the notebook editor (right side, near he submission button). This will then create dependencies which are also installed duri g submission.</p>
