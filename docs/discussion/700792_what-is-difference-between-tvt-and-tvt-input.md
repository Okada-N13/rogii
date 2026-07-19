# What is difference between 'TVT' and 'TVT_input' 

- 投稿者: houzeyu2683
- 投稿日時: 2026-05-18 05:08:56.912000
- 投票数: 2
- コメント数: 4（取得数: 4）
- トピックID: `700792`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/700792](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/700792)

## 本文

<p>hello, 
I check the <code>test/</code> folders, I find the <code>XXXXX__horizontal_well.csv</code> in the <code>train/</code> too.
Then I check the 'TVT_input' and 'TVT' are the same, and there are zero missing value in 'TVT'.
Is it the data-leak because I can copy the 'TVT' from <code>train/</code>.
Or … did I miss somthing?</p>

## コメント

### コメント 1 — PatrickAIForFun

- 投稿日時: 2026-05-18 06:30:45.483000
- 投票数: 1
- コメントID: `3459700`

<p>It is not a data leak - during inference on the actual hidden dataset you will be presented with wells which do not exist in the train set. Thus during inference you want be able to copy TVT and all you have is TVT_input.</p>

#### コメント 1.1 — houzeyu2683

- 投稿日時: 2026-05-18 07:41:50.820000
- 投票数: 0
- コメントID: `3459731`

<p>Thanks, so the problem is training on a pool of wells, then infer on another wells.</p>

#### コメント 1.2 — Nguyễn Trường Sơn

- 投稿日時: 2026-07-02 08:20:57.743000
- 投票数: 0
- コメントID: `3486160`

<p>Hi,</p>
<p>I'm new to this kind of Forecasting problem like this, so the question might be seem naive.  I'm confused of how <code>TVT_input</code> was given. As I understand, the problem in this is to predict TVT when a well dig deeper over time. For example, we have step-1 and step-2, then make use of it to predict step-3. But what about step-4, do we use the actual data from step-3 to predict it, or using step-3 prediction to keep guessing regressively (kind of like LLMs are doing)?</p>
<p>For training-set, we have the <code>TVT_input</code> exactly the same as <code>TVT</code>, so I consume it will be shifted for some rows, so that you can predict the actual <code>TVT</code> sooner a few feet. But the testing-set is also giving <code>TVT_input</code>. Idk if:</p>
<ol>
<li>Is the <code>TVT_input</code> given in hidden test set?</li>
<li>If it is, can we use it as a valid input? Or using former prediction in place of <code>TVT_input</code> to keep predicting.</li>
</ol>
<p>I know it kind of Chaos Theory things when trying to predicting deeper too much. But if they just give me feet higher, then wanting the prediction just below a bit, what is the meaning of all this? </p>
<p>Hope for your reply, if it might not too late.</p>

### コメント 2 — PC Jimmmy

- 投稿日時: 2026-05-18 05:42:03.850000
- 投票数: 0
- コメントID: `3459669`

<p>It was a big miss.  Read a number of the discussion posts.  It can be a tiny bit confusing - it's not a data leak.</p>
