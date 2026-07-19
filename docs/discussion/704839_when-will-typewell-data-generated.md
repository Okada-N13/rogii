# When will Typewell data generated?

- 投稿者: doteeee
- 投稿日時: 2026-06-06 14:27:33.907000
- 投票数: 5
- コメント数: 5（取得数: 5）
- トピックID: `704839`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/704839](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/704839)

## 本文

<p>I understood, logs from Horizontal well got generated from while drilling. As the drill starts vertically and move horizontally, we can get information regarding geological formation across.
Logs generated -> coordinate(x,y,z)  and GR , TVT </p>
<p>But  i am not able to understand when will Typewell information will get generated. Require some clarification regarding this, to understand if GR, TVT from typewell can be better used relate to TVT of horizontal well.</p>
<p>Thanks</p>

## コメント

### コメント 1 — PC Jimmmy

- 投稿日時: 2026-06-06 16:22:39.780000
- 投票数: 4
- コメントID: `3467570`

<p>Vertical type wells exist before the drilling starts - so full data set should be present at test when the scoring run starts.  </p>
<p>There are a limited number of different vertical holes in the train data - I have found that 57 wells can represent all the typewells for the 773 horizontals.  Would not be surprising to me that the test wells will use these existing verticals, but have made no attempts to confirm that.  As I found the 57 number it was also apparent that the vertical hole locations will not be a consistent distance or direction from the horizontal well, so TVT quality will vary with that distance variation.</p>

#### コメント 1.1 — doteeee

- 投稿日時: 2026-06-06 17:30:21.193000
- 投票数: 0
- コメントID: `3467585`

<p>thanks, <a href="https://www.kaggle.com/pcjimmmy">@pcjimmmy</a>. so the typewell are the vertical drilling happened somewhere near to horizontal well even before work starts to estimate the geology. is my understanding correct?</p>

##### コメント 1.1.1 — PC Jimmmy

- 投稿日時: 2026-06-07 02:05:31.240000
- 投票数: 3
- コメントID: `3467658`

<p>Yes - some of them also would be capable of oil or gas if the layer thick enough or perhaps some lease issues preventing the folks from going horizontal, but generally low ROI to drill them.  So my finding only 57 unique seems reasonable, and probably on the high side.  I think the host has mentioned in a post someone that some might be synthetic  (not the word he used).</p>

##### コメント 1.1.2 — PC Jimmmy

- 投稿日時: 2026-06-07 02:11:11.293000
- 投票数: 3
- コメントID: `3467661`

<p><a href="https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698449#3455674">"pseudo-typewells"</a> is the term used rather than synthetic.</p>

##### コメント 1.1.3 — doteeee

- 投稿日時: 2026-06-07 04:02:20.350000
- 投票数: 0
- コメントID: `3467688`

<p>thank <a href="https://www.kaggle.com/pcjimmmy">@pcjimmmy</a> , will go through refered post</p>
