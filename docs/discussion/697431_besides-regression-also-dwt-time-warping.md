# besides regression, also dwt (time warping)! 

- 投稿者: hengck23
- 投稿日時: 2026-05-06 03:38:07.141000
- 投票数: 43
- コメント数: 36（取得数: 36）
- トピックID: `697431`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697431](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697431)

## 本文

<p>related to geosteering. Here, given a strip of MD-GR of the horizontal well, "stretch and fold" it so that it matches the TVT-GR reference of the typewell</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fb40517687f05f7ecea819763d3b150dc%2FSelection_3412.png?generation=1778261192499811&alt=media" alt=""></p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F6f36c16fad39781039af137e753996dd%2FSelection_3304.png?generation=1778038684983121&alt=media" alt=""></p>
<hr>
<p>but, since the test and train locations are pretty close, pure regression might just also work</p>

## コメント

### コメント 1 — hengck23

- 投稿日時: 2026-05-18 14:00:49.023000
- 投票数: 8
- コメントID: `3459981`

<p>surprise surprise surprise
there are only 69 unique typewells in train data</p>

### コメント 2 — hengck23

- 投稿日時: 2026-05-24 13:53:22.063000
- 投票数: 1
- コメントID: `3462668`

<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F455e0d537cbbf0969d560dd5100e5a19%2FSelection_3752.png?generation=1779630800094346&alt=media" alt=""></p>

### コメント 3 — hengck23

- 投稿日時: 2026-05-17 21:14:20.227000
- 投票数: 4
- コメントID: `3459463`

<p>suddenly, i think of folding protein solution: alphafold. here we are essentially folding the horizontal GR signal. each possible trajectory is a confomer. typewell and neigbhours GR are "binding sites". Let's called it alphaSteer</p>

### コメント 4 — hengck23

- 投稿日時: 2026-05-16 01:58:36.620000
- 投票数: 3
- コメントID: `3458503`

<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F3357765646b064976681c3c8410ff588%2FSelection_3564.png?generation=1778896502903825&alt=media" alt=""></p>
<p>Real-time forward modeling and inversion of logging-while-drilling electromagnetic measurements in horizontal wells</p>

### コメント 5 — hengck23

- 投稿日時: 2026-05-18 20:03:12.817000
- 投票数: 1
- コメントID: `3460407`

<p>data augmentation<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F08d9018ad30f9230de44a25e038465e0%2FSelection_3670.png?generation=1779134589932757&alt=media" alt=""></p>

### コメント 6 — Tom

- 投稿日時: 2026-05-17 02:13:24.160000
- 投票数: 1
- コメントID: `3458940`

<p>Any Physics-informed ML approaches discovered? <a href="https://www.kaggle.com/hengck23">@hengck23</a> </p>

#### コメント 6.1 — hengck23

- 投稿日時: 2026-05-17 02:21:45.807000
- 投票数: 5
- コメントID: `3458942`

<p>issue is not physics modeling. you can verify your results with forward differentiable model:</p>
<pre><code>error = observed GR - generated GR = typwell GR (torch inteploated tvt as lookup index)
</code></pre>
<p>issue is inverse problem. many possible solutions  and we have to learn the data "preferred solution" and not the best solution.</p>
<p>it is like given 2 points A,B on map, predict how to get from A to B. there are some rules, but not strictly followed … </p>

### コメント 7 — hengck23

- 投稿日時: 2026-05-16 18:34:00.233000
- 投票数: 1
- コメントID: `3458833`

<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fef1fce03482eca082114a0817f89a6fd%2FSelection_3613.png?generation=1778956371898934&alt=media" alt=""></p>
<p>neighbour can help! e,g, they tell you the range of horizontal tvt</p>

#### コメント 7.1 — PatrickAIForFun

- 投稿日時: 2026-05-16 19:58:40.200000
- 投票数: 1
- コメントID: `3458871`

<p>You are showing/comparing the typewell logs, correct? If yes, then this is expected an has already been found (although not shifted matches, but exact matches): <a href="https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698449">https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698449</a></p>

##### コメント 7.1.1 — PC Jimmmy

- 投稿日時: 2026-05-26 15:42:49.200000
- 投票数: 1
- コメントID: `3463324`

<p>If you look for shifted matches I found only 57 type wells in the entire field.</p>

### コメント 8 — hengck23

- 投稿日時: 2026-05-16 02:07:27.100000
- 投票数: 1
- コメントID: `3458504`

<p><a href="https://www.rogii.com/blog/starsteer-geoassist-enhanced-eagle-ford-reservoir">https://www.rogii.com/blog/starsteer-geoassist-enhanced-eagle-ford-reservoir</a>
ROGII implemented StarSteer's ML-based GeoAssist to automate geosteering.</p>
<p>which parts are the most and least confident? what are ML looking at?</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F0517074d09fcb4213950a5daaa3b75f6%2FSelection_3566.png?generation=1778897341430331&alt=media" alt=""></p>

#### コメント 8.1 — hengck23

- 投稿日時: 2026-05-16 02:20:46.317000
- 投票数: 1
- コメントID: `3458506`

<p>where are the typewells? how is the global dip related to the xy slant horizontal drill path?</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fba921864d6ffa56e24368627be5574d1%2FSelection_3568.png?generation=1778898557640166&alt=media" alt=""></p>

#### コメント 8.2 — hengck23

- 投稿日時: 2026-05-16 02:28:29.603000
- 投票数: 1
- コメントID: `3458508`

<p>their heatmap looks very good (heatmap is some similarity between horizontal GR and reference geology?)</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F834455cdaf83bc4e0ba26e6670934287%2FSelection_3567.png?generation=1778898507823700&alt=media" alt=""></p>
<p><a href="https://www.rogii.com/blog/the-hidden-cost-of-switching-between-geoscience-tools">https://www.rogii.com/blog/the-hidden-cost-of-switching-between-geoscience-tools</a><br>
how wells are planned</p>

### コメント 9 — hengck23

- 投稿日時: 2026-05-17 10:00:33
- 投票数: 2
- コメントID: `3459070`

<p>Someone should probe the hidden typewell to see if they are offset copies of train. I think some are. If they are, you have free geology infotmation copied from train</p>

#### コメント 9.1 — PatrickAIForFun

- 投稿日時: 2026-05-17 11:41:15.740000
- 投票数: 1
- コメントID: `3459148`

<p>I can neither confirm nor deny with full certainty, however based on my observations and testing it also seems very likely that the hidden test set shares some typewells with the public training set.
However, apart from the geology labels of the typewell, this does not provide much more information. Furthermore ,in a real world application these topsets for the typewell would also most likely be available.  Thus, this does not compromise the general goal of the competition in my opinion.</p>
<p><a href="https://www.kaggle.com/igorakuvaev">@igorakuvaev</a> is this intended/known?</p>

##### コメント 9.1.1 — hengck23

- 投稿日時: 2026-05-17 12:06:15.220000
- 投票数: 1
- コメントID: `3459158`

<p>you can check the host competition PPT. he shows the hidden test well location</p>

##### コメント 9.1.2 — hengck23

- 投稿日時: 2026-05-17 12:07:36.930000
- 投票数: 1
- コメントID: `3459159`

<p>"apart from the geology labels of the typewell, this does not provide much more information" you are wrong.
the model now become tvt = model(shared type well, known tvt, full tvt of neighbours (include site geology and gr) )</p>

##### コメント 9.1.3 — Tom

- 投稿日時: 2026-05-17 13:47:30.753000
- 投票数: 2
- コメントID: `3459204`

<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4310004%2F53b4b42c9e20e0b927c4868cbc9ae1fd%2F532.png?generation=1779025623576570&alt=media" alt=""></p>
<p>With these information and referencing the typewells, you can even build a very strong transformer or GNN to encode them </p>

##### コメント 9.1.4 — PC Jimmmy

- 投稿日時: 2026-05-26 15:40:38.063000
- 投票数: 0
- コメントID: `3463323`

<p><a href="https://www.kaggle.com/hengck23">hengck23</a></p>
<p>When you checked the ppt did you end up with 45 test well paths?</p>

### コメント 10 — hengck23

- 投稿日時: 2026-05-08 04:46:07.447000
- 投票数: 2
- コメントID: `3454874`

<p>plot in 3d and it is a folding problem</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fa5cbb405fd3a89da3dd64ea731d85478%2FSelection_3390.png?generation=1778215560856015&alt=media" alt=""></p>

#### コメント 10.1 — hengck23

- 投稿日時: 2026-05-08 04:47:47.360000
- 投票数: 1
- コメントID: `3454876`

<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fc8c6eeef5151aab214e5273363907b35%2FSelection_3391.png?generation=1778215665544705&alt=media" alt=""></p>

### コメント 11 — hengck23

- 投稿日時: 2026-05-16 01:43:18.277000
- 投票数: 2
- コメントID: `3458497`

<p>related:</p>
<p>related:
<a href="https://github.com/hhschumann/LWD_inversion">https://github.com/hhschumann/LWD_inversion</a><br>
"This project aims to use gamma ray loging while drilling (LWD) measurements to invert for the position of a geologic interval relative to the wellbore"
<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F183302a3299a70acb01e682f43f45de6%2FSelection_3562.png?generation=1778895793013734&alt=media" alt=""></p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fd9c9158c49c5a610b3bfce0a6def03c0%2FSelection_3563.png?generation=1778895891144447&alt=media" alt=""></p>

### コメント 12 — hengck23

- 投稿日時: 2026-05-06 11:01:23.030000
- 投票数: 2
- コメントID: `3453985`

<p>forward model?</p>
<pre><code>hfile = "0a57a29c__horizontal_well.csv"
tfile = "0a57a29c__typewell.csv"
h  = pd.read_csv(f"{KAGGLE_DIR}/train/{hfile}")
tw = pd.read_csv(f"{KAGGLE_DIR}/train/{tfile}")
tw_tvt = tw["TVT"].values
tw_gr = tw["GR"].values
h_tvt = h["TVT"].values
h_gr = h["GR"].values
query_gr = np.interp(
   h_tvt,
   tw_tvt,
   tw_gr,
)
</code></pre>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F4690deb3957ae578318354dcd4385283%2FSelection_3332.png?generation=1778085488837342&alt=media" alt=""></p>

### コメント 13 — hengck23

- 投稿日時: 2026-05-06 03:51:13.910000
- 投票数: 2
- コメントID: `3453849`

<p><a href="https://github.com/luthfigeo/DTW-Stratigraphic-Correlation/blob/main/DTW.ipynb">https://github.com/luthfigeo/DTW-Stratigraphic-Correlation/blob/main/DTW.ipynb</a>
<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F8653a6175531f307e6d3f22c4b95b560%2FSelection_3305.png?generation=1778039471463012&alt=media" alt=""></p>

### コメント 14 — hengck23

- 投稿日時: 2026-05-06 10:59:03.640000
- 投票数: 3
- コメントID: `3453983`

<p>the trick to winning is to "somehow" reconstruct the "3d geological site" using the train AND test data, since the wells are in the same "site"</p>

### コメント 15 — hengck23

- 投稿日時: 2026-05-06 10:51:46.880000
- 投票数: 3
- コメントID: `3453980`

<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F7418e3be8c2b50fe18e87e7f8beb00e4%2FSelection_3321.png?generation=1778064703636922&alt=media" alt=""></p>

### コメント 16 — hengck23

- 投稿日時: 2026-05-09 18:17:55.963000
- 投票数: 2
- コメントID: `3455578`

<p>deep net logit:  horizontal md length x location of reference (each location is a class)</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F3df800b866b035d8ea3668f479cd241b%2FSelection_3456.png?generation=1778350565993562&alt=media" alt=""></p>
<p>training iterations of the transformer:</p>
<ul>
<li>it figures out the best way is to grow from PS?  I actually expect it to find anchor points first</li>
</ul>

### コメント 17 — hengck23

- 投稿日時: 2026-05-09 05:43:53.623000
- 投票数: 1
- コメントID: `3455328`

<p>plot of md vs dTVT</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F463aaa87facc21c28d172fec7a016510%2FSelection_3418.png?generation=1778305370654718&alt=media" alt=""></p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fd1dace62c505b33b21d01104722c3bf8%2FSelection_3419.png?generation=1778305406732462&alt=media" alt=""></p>
<p>this tells you how the ground truth is created … reverse engineering?</p>

#### コメント 17.1 — hengck23

- 投稿日時: 2026-05-09 07:49:03.390000
- 投票数: 3
- コメントID: `3455358`

<p>why is dy constant? synthetic data????</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F1bc50ab50551bbe3621be961745da6de%2FSelection_3427.png?generation=1778312927936006&alt=media" alt=""></p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Ffc1b73c99aa3ca582d498619a119804f%2FSelection_3428.png?generation=1778312940805533&alt=media" alt=""></p>

### コメント 18 — hengck23

- 投稿日時: 2026-05-07 04:52:16.537000
- 投票数: 1
- コメントID: `3454411`

<p>piece-wise fitting DTW.</p>
<p>model predict (start,end, dTVT/dMD slope) for each segment.</p>
<p>but i think the original DP in cost matrix is better </p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F23521faa598aa443f46d761cc1f5f7c0%2FSelection_3349.png?generation=1778129487699553&alt=media" alt=""></p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fd0bdc08a78a3bb3360082ab1e9276493%2FSelection_3362.png?generation=1778140477307972&alt=media" alt=""></p>
<ul>
<li>When the drill moves a long distance in MD / XYZ, it may still stay inside almost the same geological position.  </li>
<li>That makes alignment hard because there is little “geological movement” to match  </li>
</ul>

#### コメント 18.1 — eugene

- 投稿日時: 2026-05-07 10:59:17.863000
- 投票数: 0
- コメントID: `3454493`

<p>Do I understand you method correct? 
You move a window along the hw_gr, after ps point. For each window, you find the closest window on tw_gr with DTW. Then you look at which TVT is closest to the window on tw_gr -> this TVT  is predict ? I did that way, but it totaly not worked for me 🤔</p>

##### コメント 18.1.1 — hengck23

- 投稿日時: 2026-05-07 11:46:55.897000
- 投票数: 3
- コメントID: `3454511`

<p>It is not the normal dtw. The index can be reversed depending if the drillhead is travelling upwards or downward. The noise is quite large, maybe you need to restrict to local search.</p>
<p>The host post had a YouTube video link on rogii geosteer, that will clear things up</p>

##### コメント 18.1.2 — hengck23

- 投稿日時: 2026-05-07 12:32:34.120000
- 投票数: 0
- コメントID: `3454533`

<p><a href="https://www.kaggle.com/evgeny000">@evgeny000</a> </p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F09eb08a581fad7ca45347e14199bab63%2FSelection_3366.png?generation=1778157143610840&alt=media" alt=""></p>

##### コメント 18.1.3 — eugene

- 投稿日時: 2026-05-07 13:01:01.020000
- 投票数: 0
- コメントID: `3454554`

<p>Thanks for the explanation! I still don't fully understand the data yet 😬, I hope it will be more clear after watching the video you mentioned. As far as I understand, you don't use tw data? </p>
<p>The most confusing part to me  the tw data. The TVT in the tw data is a monotonous straight line, although as far as I understand, when different layers intersect during drilling, the thickness of the different layers changes, and the TVT should be fluctuate 🤔</p>

##### コメント 18.1.4 — hengck23

- 投稿日時: 2026-05-07 13:09:09.677000
- 投票数: 3
- コメントID: `3454560`

<p>You should think of it like that: reference vertical typewell has gr that encodes the geologic location called tvt. We are in horizontal well with unknown location. We want to know what is our tvt. But we have signal gr. We need to move up or down to have generate enough gr signal signature for matching reference gr to guess the tvt.  Tvt is location relative to the geology layer we want to be.  </p>

### コメント 19 — Navneet

- 投稿日時: 2026-05-10 05:38:00.933000
- 投票数: -1
- コメントID: `3455713`

<p>Cool info on geosteering <a href="https://www.kaggle.com/hengck23">@hengck23</a> </p>
