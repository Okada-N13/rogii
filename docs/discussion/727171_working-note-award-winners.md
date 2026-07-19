# Working Note Award - winners!!!

- 投稿者: Igor Kuvaev
- 投稿日時: 2026-07-18 03:51:39.763000
- 投票数: 23
- コメント数: 0（取得数: 0）
- トピックID: `727171`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/727171](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/727171)

## 本文

<p>We've picked two winners for the Working Note Award:</p>
<p><a href="https://www.kaggle.com/writeups/radiantallomancer/when-better-cv-scores-worse-a-control-first-geost">When Better CV Scores Worse</a> by <a href="https://www.kaggle.com/radiantallomancer">@radiantallomancer</a> 
<a href="https://www.kaggle.com/writeups/malyshevdanil/the-wiggle-is-free-the-trend-is-the-wall">The Wiggle Is Free the Trend Is the Wall</a>,  by <a href="https://www.kaggle.com/malyshevdanil">@malyshevdanil</a> </p>
<p>Honestly, this was a hard call. Quite a few write-ups landed in the same score band and had interesting ideas worth stealing, so raw score wasn't the deciding factor. What set these two apart was coverage and honesty: they don't just describe the winning pipeline, they show what didn't work and why, including how easy it was to overfit to the public leaderboard.</p>
<p><a href="https://www.kaggle.com/radiantallomancer">@radiantallomancer</a> 's note is the best example in the field of validation discipline. Local CV was actually anti-correlated with the public score, and instead of chasing public deltas he built controls around every claimed gain - shuffle tests, no-op checks, leave-spatial-out audits.</p>
<p><a href="https://www.kaggle.com/malyshevdanil">@malyshevdanil</a> 's note has the cleanest framing of the problem itself: the high-frequency wiggle is free (it's just the trajectory), all the error lives in the low-frequency trend. His oracle ladder shows exactly how many feet sit in datum vs slope vs shape, and he proves with a hard number why a single well's gamma can't recover the datum. Also full credit for reporting negative results plainly - the failed GR-matching experiments will save the next team a lot of time.</p>
<p>The two complement each other well: one is strongest on validation and model fusion, the other on decomposition and identifiability. Together they're pretty much a manual for anyone building a real geosteering pipeline out of this competition.</p>
<p>Thanks to everyone who submitted a write-up - there were many more good ideas than we could award. </p>
<p>Congrats to the winners!</p>

## コメント

_コメントなし_
