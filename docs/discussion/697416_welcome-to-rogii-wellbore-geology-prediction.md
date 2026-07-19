# Welcome to ROGII - Wellbore Geology Prediction!

- 投稿者: Igor Kuvaev
- 投稿日時: 2026-05-06 00:04:21.663000
- 投票数: 34
- コメント数: 15（取得数: 15）
- トピックID: `697416`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697416](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697416)

## 本文

<p>We (ROGII) are excited to welcome you to this challenge.</p>
<p>The presented dataset is a very typical example of modern horizontal drilling data. It contains the well trajectory along with Gamma Ray (GR) measurements collected along the wellbore. This dataset can be confidently interpreted by an experienced human interpreter.</p>
<p>We look forward to the results of this competition, as they may help enable automation during the well drilling process.</p>
<p>We strongly encourage you to <strong>review the PowerPoint presentation attached to the competition data</strong>, as it contains key information on horizontal well interpretation.</p>
<p>Additionally, feel free to watch how the manual geosteering process works:
<a href="https://youtu.be/VgzFt7xknGo?si=LfqTlmjsZpNmtbJH&t=1034">https://youtu.be/VgzFt7xknGo?si=LfqTlmjsZpNmtbJH&t=1034</a></p>
<p>Most importantly, make sure you enjoy the competition. You are solving a real-world problem, and successful solutions will be greatly appreciated by the industry by making drilling safer, more efficient, and lower in emissions.</p>
<p>If you have any questions, we would be happy to address them.</p>
<p>— On behalf of all organizers,
ROGII</p>

## コメント

### コメント 1 — Tiago Soares

- 投稿日時: 2026-06-04 23:54:37.957000
- 投票数: 11
- コメントID: `3466924`

<p>Hi <a href="https://www.kaggle.com/igorakuvaev">@igorakuvaev</a> the youtube video is private, can't access it</p>

#### コメント 1.1 — Igor Kuvaev

- 投稿日時: 2026-06-09 21:56:25.080000
- 投票数: 1
- コメントID: `3468835`

<p>Appologies, the link in the original post has been updated.</p>

### コメント 2 — Tom

- 投稿日時: 2026-05-17 11:45:44.230000
- 投票数: 3
- コメントID: `3459151`

<p>Hi <a href="https://www.kaggle.com/igorakuvaev">@igorakuvaev</a>, do we allow to access starsteer software?</p>

#### コメント 2.1 — Igor Kuvaev

- 投稿日時: 2026-06-09 21:57:50.867000
- 投票数: 0
- コメントID: `3468837`

<p>no, StarSteer will not help you much as it just manually projects GR onto TVT scale based on human interpretation. You can quickly code this by yourself with the training dataset.</p>

### コメント 3 — Sharmi Nisha

- 投稿日時: 2026-05-24 00:47:37.340000
- 投票数: 2
- コメントID: `3462577`

<p>Hi <a href="https://www.kaggle.com/igorakuvaev">@igorakuvaev</a>, How should we use typewell GR to predict TVT after PS point? Is DTW the right approach or something else?"</p>

#### コメント 3.1 — Igor Kuvaev

- 投稿日時: 2026-06-09 21:56:47.693000
- 投票数: 0
- コメントID: `3468836`

<p>the answer to this question is the goal of the competition</p>

##### コメント 3.1.1 — BIT_Guber

- 投稿日時: 2026-07-13 06:06:15.343000
- 投票数: -1
- コメントID: `3496017`

<p>Hi <a href="https://www.kaggle.com/igorakuvaev">@igorakuvaev</a>, <a href="https://www.kaggle.com/macruzbar">@macruzbar</a> <a href="https://www.kaggle.com/ryanholbrook">@ryanholbrook</a> , it seems top public notebooks most depend on geolocation features instead of independent from it, since test wells locations reveal in pptx file and it too close. i tried myself only location based solution 10.5 but independent got 13.6. i noticed neighbor wells similar trajectory( z/tvt ). can guys add some random noise in test coordinate it help generalization solution in Leaderboard 😟</p>

### コメント 4 — Surena

- 投稿日時: 2026-05-11 07:52:13.477000
- 投票数: 2
- コメントID: `3456095`

<p>Hi <a href="https://www.kaggle.com/igorakuvaev">@igorakuvaev</a> and <a href="https://www.kaggle.com/macruzbar">@macruzbar</a>,</p>
<p>I have a question regarding the competition eligibility rules.</p>
<p>If a team wins a prize, but one of the team members is not eligible to receive prizes (for example, due to being under 18 or residing in a restricted country), while the team leader and other members are eligible, how would the prize distribution and eligibility be handled in that case?</p>
<p>Would the ineligible participant simply be excluded from receiving the prize, or could this affect the eligibility of the entire team?</p>
<p>Thank you!</p>

#### コメント 4.1 — María Cruz

- 投稿日時: 2026-05-26 23:41:23.593000
- 投票数: 4
- コメントID: `3463463`

<p>Hi <a href="https://www.kaggle.com/surenasarvari">@surenasarvari</a> -- it depends on why the participant may not be eligible. If it is a restricted country, the individual is excluded, and the prize is split among the remaining team members. If it is a rule-based ineligibility, for example, an employee of the competition entity, etc, the entire team is ineligible. If the participant is under 18 years old, they need to submit a form where their parents give consent to the participant joining the competition. </p>
<p>Let me know if you need more information for participants under 18 years old. Happy to help, and good luck!</p>
<p>María</p>

### コメント 5 — Iambruce

- 投稿日時: 2026-06-09 04:49:55.103000
- 投票数: 0
- コメントID: `3468467`

<p>Hello <a href="https://www.kaggle.com/igorakuvaev">@igorakuvaev</a> Youtube video is private, how can we get the video access?</p>

#### コメント 5.1 — Igor Kuvaev

- 投稿日時: 2026-06-09 21:55:39.213000
- 投票数: 1
- コメントID: `3468834`

<p>Appologies, the youtube link was corrupted. Updated the link in the original post.
Thank you</p>

### コメント 6 — Unknown

- 投稿日時: 2026-05-27 06:28:38.787000
- 投票数: 0
- コメントID: `3463528`

_本文なし_

#### コメント 6.1 — PatrickAIForFun

- 投稿日時: 2026-05-27 06:33:54.280000
- 投票数: 1
- コメントID: `3463530`

<p>Remember that the test data you download is just a partial copy of the training set to test your inference code. The true hidden test set is only provided to your code after submitting and does not have any overlap with the training et (except for the typewells -> which is known and to be expected)</p>

### コメント 7 — Unknown

- 投稿日時: 2026-05-24 00:40:14.500000
- 投票数: 0
- コメントID: `3462575`

_本文なし_

### コメント 8 — Aditi Chatterjee

- 投稿日時: 2026-06-07 04:00:49.780000
- 投票数: 2
- コメントID: `3467685`

<p>Thanks for creating this challenge!</p>
