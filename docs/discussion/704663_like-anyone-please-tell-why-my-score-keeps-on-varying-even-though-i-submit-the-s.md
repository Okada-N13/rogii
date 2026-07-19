# Like anyone please tell why my score keeps on varying even though I submit the same notebook?

- 投稿者: Debatreya Biswas
- 投稿日時: 2026-06-05 17:00:25.712000
- 投票数: 9
- コメント数: 6（取得数: 6）
- トピックID: `704663`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/704663](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/704663)

## 本文

<p>like this is is the notebook I am talking about <a href="https://www.kaggle.com/code/debatreyabiswas/wellboregeology-prediction-with-koolbox-best-8-188">https://www.kaggle.com/code/debatreyabiswas/wellboregeology-prediction-with-koolbox-best-8-188</a></p>
<p>I submit it 3 times and all the 3 times different scores were given</p>
<p>like first time 8.354</p>
<p>second time 8.188(the best one)</p>
<p>third time 8.438</p>
<p>like is it because of the noise in the data we are using?</p>
<p>like I know the difference is only around 0.2 but when the models in the leaderboard board are only separated by 0.01 in some cases it becomes a huge deal, so it is just luck? Or is there a legit solution to this problem other than hoping for the best</p>
<p>(Sorry, I am new to Kaggle so the question might be stupid)</p>

## コメント

### コメント 1 — Chris Deotte

- 投稿日時: 2026-06-05 21:10:30.897000
- 投票数: 3
- コメントID: `3467297`

<p>Many feature engineering are stochastic in this competition. Feature engineering is the process of us making new columns on the train and test data. So every time we submit our notebook, the train.csv will be different and the test.csv will be different because when we add our features they are different (stochastic) each time.</p>
<p>(And train.csv affects the trained model and test.csv affects the inference of the model)</p>

#### コメント 1.1 — Debatreya Biswas

- 投稿日時: 2026-06-05 21:32:50.960000
- 投票数: 2
- コメントID: `3467309`

<p>Ok,thank you Sir,</p>
<p>So,the stochastic is the model randomness,(like for example while making a cake there is a instructions given add a pinch of salt in my code,so whenever I submit it,kaggle add different values to that pinch of salt command everytime leading to different tasting cake,here which is different scores).</p>
<p>So,Sir,does averaging different randomness values helps,to get a much more stable version of this model?</p>

##### コメント 1.1.1 — Chris Deotte

- 投稿日時: 2026-06-05 22:17:50.310000
- 投票数: 6
- コメントID: `3467317`

<p>No random is like random. In other tabular data competitions, there are no random features.</p>
<p>In this competition we do things like particle filter features. To create a new column we literally generate a bunch of random numbers and then compute stuff from those random numbers. So the resultant new columns is a result of randomness. If we were to compute the new column twice it would be different each time.</p>
<p>This is completely different from making a new column that is the subtraction of two previous columns. That is how most feature engineering is. Most feature engineering is a deterministic formula and if we were to create the new column twice it would have the same values each time.</p>

##### コメント 1.1.2 — Debatreya Biswas

- 投稿日時: 2026-06-05 23:00:09.223000
- 投票数: 2
- コメントID: `3467335`

<p>Wow,ok understood Sir🤯</p>
<p>So,everytime we are submitting it is like a full on random roll of dice to generate the new column.</p>
<p>So, basically there can exist a infinite possible number between my given range of let's say in this case 8.1 to 8.5,right Sir?</p>

##### コメント 1.1.3 — Chris Deotte

- 投稿日時: 2026-06-05 23:04:44.600000
- 投票数: 4
- コメントID: `3467337`

<p>Yes exactly. A few weeks ago, one could get top 5 on the public LB (not private LB) just submitting the best public notebook like 10 times haha. But not anymore because the people at the top of the leaderboard are doing more than just the best public notebooks at this point.</p>

##### コメント 1.1.4 — Debatreya Biswas

- 投稿日時: 2026-06-05 23:18:51.757000
- 投票数: 1
- コメントID: `3467340`

<p>Lol 😂 just praying for luck to get the top spot</p>
<p>Thank,you Sir so much for the explanation and helping me understand it all</p>
<p>Btw, Sir as you said the people at the top are now doing things differently do you have any hints on topics that I should learn or apply in these models to reach the top of the leaderboard?</p>
