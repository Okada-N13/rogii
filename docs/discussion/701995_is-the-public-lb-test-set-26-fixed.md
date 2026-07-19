# Is the public LB test set (26%) fixed? 

- 投稿者: Alhasan Abdellatif
- 投稿日時: 2026-05-20 15:55:07.098000
- 投票数: 9
- コメント数: 13（取得数: 13）
- トピックID: `701995`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/701995](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/701995)

## 本文

<blockquote>
  <p>This leaderboard is calculated with approximately 26% of the test data. The final results will be based on the other 74%, so the final standings may be different.</p>
</blockquote>
<p>I have noticed that running & submittin the same exact notebook, same trained models, gives very different score with differences reaching over ~ 0.5 ft.  what does this mean? Is the public LB test set (26%) fixed?  </p>

## コメント

### コメント 1 — Chris Deotte

- 投稿日時: 2026-06-05 21:11:38.093000
- 投票数: 6
- コメントID: `3467298`

<p>The test data does not change.</p>
<p>The reason our scores change is because many feature engineering are stochastic in this competition. Feature engineering is the process of us making new columns on the train and test data. So every time we submit our notebook, the train.csv will be different and the test.csv will be different because when we add our features they are different (stochastic) each time.</p>
<p>(And train.csv affects the trained model and test.csv affects the inference of the model)</p>

### コメント 2 — Zhenyu Zhang

- 投稿日時: 2026-05-21 13:13:24.200000
- 投票数: -1
- コメントID: `3461757`

<p>I think it is not fixed</p>

### コメント 3 — Hamza

- 投稿日時: 2026-05-23 10:56:46.810000
- 投票数: 0
- コメントID: `3462446`

<p>I run my same LGB Baseline 2 times, 1st time I got score of 9.964 and Second time I got the score 9.477</p>

### コメント 4 — YtLiu

- 投稿日時: 2026-05-22 06:51:17.183000
- 投票数: 0
- コメントID: `3462108`

<p>The test set should be fixed. The score discrepancy you’re experiencing is most likely due to randomness in your code not being fully controlled. Multi-process parallelism and GPU usage can both introduce randomness. Additionally, if you’re using Numba for acceleration, the seed set at the Python level won’t be propagated to the functions compiled by Numba. When I tried to completely control the randomness, the scores for the two submissions were consistent.</p>

### コメント 5 — PC Jimmmy

- 投稿日時: 2026-05-20 17:48:37.077000
- 投票数: 0
- コメントID: `3461502`

<p>As noted by <a href="https://www.kaggle.com/patrickaiforfun">PatrickAIForFun</a> - the test has never varied in the 8 years I have been here.</p>
<p>Not sure I understood your difference value - what is the smallest and largest score you have for what you believe was the exact same notebook?</p>

#### コメント 5.1 — Alhasan Abdellatif

- 投稿日時: 2026-05-20 19:11:28.777000
- 投票数: 0
- コメントID: `3461536`

<p>For example, copying and re-submitting this top scored public notebook <a href="https://www.kaggle.com/code/nihilisticneuralnet/9-251-rogii-wellbore-geology-prediction-dwt-based/notebook">https://www.kaggle.com/code/nihilisticneuralnet/9-251-rogii-wellbore-geology-prediction-dwt-based/notebook</a> led to a 9.724 which is around 0.5 ft difference than its recorded score 9.251, also another submission scored 9.962. This also happens with some of my own notebooks. I will double check the randomness and see. Thanks!</p>

##### コメント 5.1.1 — PC Jimmmy

- 投稿日時: 2026-05-21 12:30:01.137000
- 投票数: 0
- コメントID: `3461752`

<p>Copied and re-submitted the notebook and will let you know in few hours how it scored for me.  But as noted results do vary even with a very detailed seed, but 0.5 does seem a bit on the high side.  I would assume that kaggle might be running your code on different data center than mine.  </p>

##### コメント 5.1.2 — PC Jimmmy

- 投稿日時: 2026-05-21 15:00:40.743000
- 投票数: 1
- コメントID: `3461814`

<p>WOW - I did even worse at 10.146.   </p>

##### コメント 5.1.3 — PC Jimmmy

- 投稿日時: 2026-05-21 15:09:17.487000
- 投票数: 0
- コメントID: `3461818`

<p>My LGB model rmse values match the posted original code.
My Catboost values also match the posted orginal code.
My Running Hill Climbing values match.
My predicted values for the fake test data don't match - 
11747.366412 vs 11747.366702 for the very first for example.</p>
<p>This seems more like a code error/omission rather than a seeding issue.</p>

##### コメント 5.1.4 — PC Jimmmy

- 投稿日時: 2026-05-22 14:42:33.990000
- 投票数: 0
- コメントID: `3462218`

<p>In the original notebook that you forked the code from there is a comment from at least one other person who got a different value despite using the exact code.</p>

### コメント 6 — PatrickAIForFun

- 投稿日時: 2026-05-20 16:15:43.110000
- 投票数: 0
- コメントID: `3461464`

<p>This most likely means that not all randomness is fixed within your notebook (sometimes, even fixing all random seeds is not deterministic when using the GPU).
The 26% public split is fixed (not a host - thus can't confirm 100% but it wouldn't make sense to vary it and I don't know of any competition where this would have been the case).</p>

#### コメント 6.1 — Alhasan Abdellatif

- 投稿日時: 2026-05-20 19:06:18.230000
- 投票数: 0
- コメントID: `3461533`

<p>Completely agree. It does not make sense if it vaires. I will double check the randomness in the notebook. Thanks!</p>

##### コメント 6.1.1 — Radmir Zosimov

- 投稿日時: 2026-05-21 13:28:41.353000
- 投票数: 0
- コメントID: `3461763`

<p>I had the same issue, it’s most likely your feature generation includes randomness, fix your seed. Also if you use numba seed has to be set inside a function </p>
