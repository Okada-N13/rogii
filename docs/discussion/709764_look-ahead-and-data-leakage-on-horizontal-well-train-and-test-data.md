# Look Ahead and Data Leakage on Horizontal Well Train and Test Data

- 投稿者: Tiago Soares
- 投稿日時: 2026-06-19 21:53:51.377000
- 投票数: 1
- コメント数: 6（取得数: 6）
- トピックID: `709764`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/709764](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/709764)

## 本文

<p>Hi organizers, apologies if this is a repeated question, would you be able to clarify if by making our model use feature data such GR and known future trajectory of the horizontal drill from rows that are ahead of the predicted row considered invalid/data leakage for the purpose of this competition?</p>
<p>From what I've seen on other competitions and discussions on this one, I believe this is relevant topic for many of the competitors. Thank you for your time</p>

## コメント

### コメント 1 — Tiago Soares

- 投稿日時: 2026-06-20 12:01:30.810000
- 投票数: 0
- コメントID: `3476225`

<p>At the end of the competition, in order to win the prize we'll need to justify the solution, I believe that it will be unlikely that the prize would be provided to someone who made a solution that follows a logic that doesn't hold water in a real drill scenario.</p>
<p>If we use a custom look ahead feature with the GR signal to characterize the current row's layer I would assume it would be considered an invalid approach.</p>
<p>The reason why I ask is because I believe this is ultimately a margin that can make a difference in the final scoring</p>

#### コメント 1.1 — PatrickAIForFun

- 投稿日時: 2026-06-20 12:20:08.203000
- 投票数: 1
- コメントID: `3476242`

<p>As per my understanding there are actually two ways to use/apply this technique in a real world scenario:
1.) Live drilling - this would be such that you have to tell where you are in the geology purely based on past observations and would thus suffer from data leakage as you described.
2.) Post drilling analysis - this is what is shown in most geosteering tutorials and is done after the full well is drilled, thus using all available data is fair games. The main goal here is to get a better understanding of the locations geology to plan future wells. My guess  (based on the way this competition is set up - if the goal would be 1. then the code submission would be setup using a prediction server where we only ever get access to past data) is this is the primary goal and thus using all avaialable data is fair gamesl.</p>

##### コメント 1.1.1 — Tiago Soares

- 投稿日時: 2026-06-20 12:42:48.477000
- 投票数: 0
- コメントID: `3476273`

<p>I get your point, but putting aside the real-world scenario, this is ultimately still a test file that will have specific "test file rules" that will differ from the training file rules. I totally agree that it seem logical that we can do any sort post drilling analysis on the train file, but what would then be allowed to transpose onto the test file, what kinds of feature engineering would be allowed on the test file in order to apply our final model… That's ultimately my point</p>

##### コメント 1.1.2 — PatrickAIForFun

- 投稿日時: 2026-06-20 13:15:20.963000
- 投票数: 0
- コメントID: `3476314`

<p>Traditionally in kaggle competitions doing everything with the data we are given is fair games UNLESS explicitly forbidden.</p>

### コメント 2 — Tabish Shah Mohsin

- 投稿日時: 2026-06-20 03:00:20.123000
- 投票数: 0
- コメントID: `3475533`

<p>Based on my understanding, as you are able to submit that, it serves as an acceptable solution.</p>

#### コメント 2.1 — Tiago Soares

- 投稿日時: 2026-06-20 11:59:15.663000
- 投票数: 0
- コメントID: `3476221`

<p>I've just updated the post to also include the test data at inference time. On my side I'm not so sure, especially because there are several different ways in which you may perform data leakage. The thing is, we're just guessing without confirmation</p>
