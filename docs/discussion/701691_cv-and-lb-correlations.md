# cv and lb correlations .....

- 投稿者: Gaurav Rawat
- 投稿日時: 2026-05-19 01:24:38.821000
- 投票数: 16
- コメント数: 16（取得数: 16）
- トピックID: `701691`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/701691](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/701691)

## 本文

<p>Was seeing some notebooks getting better in LB based on the cv , So far wanted to check how others have been getting the correlation going for tabular and non tabular models . For me have been using the standard GroupKfold on <code>wells</code> . </p>
<p>Train Log </p>
<table>
<thead>
<tr>
<th>Version</th>
<th>CV RMSE (ft)</th>
<th>LB RMSE (ft)</th>
</tr>
</thead>
<tbody>
<tr>
<td>train_v2.py</td>
<td>31.3871</td>
<td>35.843</td>
</tr>
<tr>
<td>train_v2.1.py</td>
<td>14.7065</td>
<td>13.949</td>
</tr>
<tr>
<td>v2.2</td>
<td>14.4634</td>
<td>13.777</td>
</tr>
<tr>
<td>v2.5</td>
<td>11.9993</td>
<td>12.383</td>
</tr>
<tr>
<td>v2.6</td>
<td>11.2693</td>
<td>12.383</td>
</tr>
<tr>
<td>v2.7</td>
<td>10.7485</td>
<td>10.606</td>
</tr>
<tr>
<td>v2.7.1</td>
<td>10.2486</td>
<td>-</td>
</tr>
<tr>
<td>v2.8</td>
<td>10.6543</td>
<td>-</td>
</tr>
<tr>
<td>v2.7.2 online</td>
<td>-</td>
<td>10.520</td>
</tr>
<tr>
<td>v2.9</td>
<td>10.3256</td>
<td>9.816</td>
</tr>
<tr>
<td>v2.10</td>
<td>10.3730</td>
<td>10.384</td>
</tr>
<tr>
<td>v2.9.7</td>
<td>10.37</td>
<td>9.585</td>
</tr>
<tr>
<td>v2.9.11</td>
<td>10.6</td>
<td>8.739</td>
</tr>
</tbody>
</table>

## コメント

### コメント 1 — Ruby

- 投稿日時: 2026-06-08 14:30:58.543000
- 投票数: 6
- コメントID: `3468148`

<p>recent two experiments:
CV 6.74 LB 6.48
CV 6.22 LB 7.18
I guess it is dominated by some bad cases</p>

### コメント 2 — Tucker Arrants

- 投稿日時: 2026-06-08 14:23:48.600000
- 投票数: 4
- コメントID: `3468143`

<p>LB feels noisy to me. I often observe CV improvements of 0.7 feet or more, leading to regressions in LB. I think some of my submissions were "lucky" e.g. CV around 8 scoring 6.6 on LB.</p>
<p>Latest results:</p>
<p>CV 6.7, LB 6.3 -> the "lucky gap" between CV and LB I previously observed (and can be observed on all the public notebooks) is starting to shrink. Think we need to be careful with LB here. Small number of public test set wells and heavy tail problem…</p>

#### コメント 2.1 — Jack

- 投稿日時: 2026-06-08 19:34:22.513000
- 投票数: 0
- コメントID: `3468317`

<p>I'd be questioning where those CV improvements are coming from in relation to past runs - might be insightful. I'm still trying to figure out what the heck you're doing for 2 mins inference.. well, that and orbit wars lol</p>

##### コメント 2.1.1 — Gaurav Rawat

- 投稿日時: 2026-06-08 19:49:29.430000
- 投票数: 0
- コメントID: `3468326`

<p>I dunno CV strategy needs to be alteare in one I got cv 7.4 but LB is like 9 . maybe need to have cv mixed with hard wells vs easy ones per fold or some custom way </p>

##### コメント 2.1.2 — Jack

- 投稿日時: 2026-06-08 20:10:27.660000
- 投票数: 0
- コメントID: `3468336`

<p>But how would you define hard vs easy wells? I can think of some more direct ways of balancing folds</p>

### コメント 3 — Tucker Arrants

- 投稿日時: 2026-05-29 02:50:39.137000
- 投票数: 6
- コメントID: `3464167`

<p>Single model NN update:</p>
<p>CV 8.5, LB 7.5</p>
<p>Inference in 2 minutes lol</p>

#### コメント 3.1 — Gaurav Rawat

- 投稿日時: 2026-05-29 03:31:37.767000
- 投票数: 1
- コメントID: `3464172`

<p>awesome ya I see NN infer like 2-3 mins .. :) maybe u framed the right Arch .. my cv not going down </p>

### コメント 4 — Tucker Arrants

- 投稿日時: 2026-05-19 02:03:43.993000
- 投票数: 6
- コメントID: `3460777`

<p>With the plain jane GBDT models, CV around 11.00 split on well ID and leaderboard around 9.6</p>
<p>Large gap, but very stable -> all CV improvements have led to LB improvements (so far).</p>
<p>A lot of the public notebooks have leakage which leads to a smaller CV-LB gap, but is not as trustworthy.</p>

#### コメント 4.1 — Gaurav Rawat

- 投稿日時: 2026-05-19 02:07:19.763000
- 投票数: 1
- コメントID: `3460780`

<p>feel need to beat the 10 mark in cv to see marked improvements for my experiments . NN so far for me havent been doign that great maybe need to dive deep to design them better . </p>

### コメント 5 — Gaurav Rawat

- 投稿日時: 2026-05-28 17:01:38.753000
- 投票数: 1
- コメントID: `3464047`

<p><strong>Adding NN experiments now , just baselines now</strong></p>
<ul>
<li>CV 14.4 LB 17</li>
<li>cv 8 lb 9</li>
</ul>

### コメント 6 — shanzhong8

- 投稿日時: 2026-05-21 05:32:43.863000
- 投票数: 1
- コメントID: `3461673`

<p>CV 10.7  , LB 9.9</p>

#### コメント 6.1 — Gaurav Rawat

- 投稿日時: 2026-05-22 03:03:57.753000
- 投票数: 0
- コメントID: `3462054`

<p>nice GBDT or NN .. was wondering how much folks are gettign with NN CV</p>

##### コメント 6.1.1 — shanzhong8

- 投稿日時: 2026-05-23 08:53:31.673000
- 投票数: 2
- コメントID: `3462428`

<p>Transformer</p>

### コメント 7 — Hassan Gasim

- 投稿日時: 2026-05-20 06:22:27.067000
- 投票数: -3
- コメントID: `3461301`

<p>Outstanding progress so far. Regarding your note on Neural Networks underperforming: standard MLPs usually struggle with the spatial, sequential nature of wellbore data compared to GBDTs. Since well logs are essentially depth-series data, have you considered a hybrid architecture? Implementing a 1D-CNN or a light Transformer backbone (like TabNet or FT-Transformer) grouped by Well ID can capture the vertical stratigraphy layers much better than vanilla NNs. Looking forward to seeing how your NN experiments evolve once the architecture matches the geological domain!</p>

### コメント 8 — Durga Kumari

- 投稿日時: 2026-05-19 15:09:56.927000
- 投票数: -4
- コメントID: `3460997`

<p>Interesting that v2.10 had slightly worse CV but matched LB almost perfectly. Usually a good sign the model is generalizing more consistently rather than optimizing fold-specific patterns.</p>

### コメント 9 — YYH

- 投稿日時: 2026-06-09 01:26:58.327000
- 投票数: 0
- コメントID: `3468434`

<p>Are the top-ranked solutions currently all based on physics models combined with machine learning? It seems that some physics models have performed quite well in this competition.It is easy to find numerous hard samples in the dataset that cannot be predicted accurately by conventional physical models, which drives up the RMSE score. I am currently working to address this issue.</p>
