# Cluster wells by dz offset

- 投稿者: Connor Tynan
- 投稿日時: 2026-06-21 09:58:35.245000
- 投票数: 10
- コメント数: 7（取得数: 7）
- トピックID: `711308`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/711308](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/711308)

## 本文

<p>See: <a href="https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699853#3462934">https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699853#3462934</a></p>
<p>There is an observed relationship between <code>dz / dMD</code> and <code>dTVT / dMD</code>.</p>
<p>For wells with little or no dip behaviour, these quantities are almost perfectly anti-correlated, i.e. <code>corr(dz, dTVT) ≈ -1</code>.</p>
<p>However, this relationship begins to diverge as dip behaviour becomes more significant. A practical mitigation is to fit a least-squares mapping, such that <code>dTVT ≈ a * dz + b</code>, where <code>a</code> and <code>b</code> are estimated by least squares.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F8859571%2F64f98b35054cce7b5be473145db4366c%2Fcumsum_dz_dtvt_ls_015fe0d2.png?generation=1782033397118441&alt=media" alt=""></p>
<p>If we perform this routine for all wells we can examine the distributions of a and b:</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F8859571%2Fb117826e9f20b5022b4453f7b7cb90f2%2Fglobal_ls_ab_distributions.png?generation=1782034990008775&alt=media" alt=""></p>
<p>So, we have a multi-modal distribution for B. Interestingly, if we visualise all train wells on the X-Y plane and colour by which peak they're closest to, we observe a pattern:</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F8859571%2Fe8262061f31858676463bbce8bb1c8b2%2Fxy_wells_global_b_peak_class.png?generation=1782035262356731&alt=media" alt=""></p>
<p>It is also possible to cluster these wells using X-Y-Z and corresponding B-peak:</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F8859571%2Fef128ea063c7d5a8b75f951a7c4ecce5%2Fxy_wells_b_peak_clusters_tests_professional.png?generation=1782035727553855&alt=media" alt=""></p>
<p>I tried clustering test wells by X-Y-Z as well as last-300 TVT, etc., and fitting dz to dTVT using a and b parameters from local wells / cluster. Baseline result was LB ~ 12.8. </p>

## コメント

### コメント 1 — Connor Tynan

- 投稿日時: 2026-07-07 01:43:36.500000
- 投票数: 1
- コメントID: `3492388`

<p>Link to EDA notebook: <a href="https://www.kaggle.com/code/connortynan/dz-dtvt-eda">https://www.kaggle.com/code/connortynan/dz-dtvt-eda</a></p>

### コメント 2 — Ulrich G.

- 投稿日時: 2026-07-06 15:22:35.967000
- 投票数: 1
- コメントID: `3491051`

<p><a href="https://www.kaggle.com/connortynan">@connortynan</a> would you mind sharing more or a notebook for your 12.8 pipeline ? Kindly Ulrich G.</p>

#### コメント 2.1 — Connor Tynan

- 投稿日時: 2026-07-07 01:30:15.583000
- 投票数: 0
- コメントID: `3492376`

<p><a href="https://www.kaggle.com/ulrich07">@ulrich07</a> here is a link to my current working: <a href="https://www.kaggle.com/code/connortynan/rogii-k16-spline-kernel-knn-adaptive-kappa?scriptVersionId=333236752">https://www.kaggle.com/code/connortynan/rogii-k16-spline-kernel-knn-adaptive-kappa?scriptVersionId=333236752</a></p>
<p>It is in the same spirit as this discussion post, but I have moved towards using a spline instead of least squares. The score is ~10.8 LB, which correlates with my local CV. I am hoping, that with some additional EDA and tweaking, the LB can be dropped below 10 and possibly be used as a prior of sorts for either a 2D-CNN or tabular model (??).</p>

##### コメント 2.1.1 — Ulrich G.

- 投稿日時: 2026-07-08 14:44:19.497000
- 投票数: 1
- コメントID: `3493811`

<p>Thank you so much 👍. I think that your approach is underrated. </p>

##### コメント 2.1.2 — Connor Tynan

- 投稿日時: 2026-07-08 21:51:42.337000
- 投票数: 1
- コメントID: `3493986`

<p>Thank you. It's nice to see that you can do quite a lot with just the geometry! I've tried incorporating GR but have had limited success (marginal RMSE gains), which aligns with other users' experiences from what I've read.</p>

### コメント 3 — Connor Tynan

- 投稿日時: 2026-06-21 10:03:10.910000
- 投票数: 1
- コメントID: `3477158`

<p>If we simulate h_GR using TVT and typewell (<a href="https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699853#3465106">https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699853#3465106</a>) we can visualise how simulated h_GR varies by cluster:</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F8859571%2Fc8d7874f964cb9a0ea0b1c56a4520c4d%2Fhgr_cluster_similarity_cluster_overlays.png?generation=1782036155581548&alt=media" alt=""></p>
<p>GR is too noisy to establish whether there is any meaningful relationship between clustered wells.</p>

### コメント 4 — Georgy Mamarin

- 投稿日時: 2026-06-30 11:29:18.450000
- 投票数: -5
- コメントID: `3484435`

<p>This is a clean way to surface the structure — fitting dTVT ≈ a*dz + b per well and then looking at where b lives is a nice move, and the X-Y pattern in the b-peaks looks like real signal to me, not an artifact.</p>
<p>Here's why I think it behaves that way, from some train-side measurements. The TVT <em>level</em> is strongly coherent in space: a well's last-known TVT predicts its nearest neighbour's at correlation ~0.98 (on 265 wells). That's the signal your X-Y clustering of b is picking up — neighbours share a base surface. But the thing you actually have to predict on the tail, the in-zone drift (the dip away from flat), has nearest-neighbour correlation near zero — structureless from one well to the next. So a cluster hands you the offset but not the dip, which I'd guess is why the cluster-transfer lands around your ~12.8 rather than down with the GR-matching heads: it pins the level and misses the slope.</p>
<p>On your second point — "GR too noisy to establish a relationship between clustered wells" — I think it's worse than noise, and more interesting. GR(TVT) is self-similar here: the Eagle Ford is rhythmically bedded (souldrive's ±15 ft datum thread has the Milankovitch story), so the horizontal GR lines up with the typewell at two positions about one bundle apart. I measured that even an oracle registration allowed to see the true eval GR recovers the eval slope at essentially zero correlation, and souldrive found from the cost side that when there's a second minimum, the margin between the two carries no information about which is right (r ≈ +0.05). So denoising won't rescue it — the ambiguity is structural, not sensor noise.</p>
<p>If it's useful, I worked the level-vs-drift split and the bimodal datum out in more detail here: <a href="https://www.kaggle.com/code/georgymamarin/stop-reforking-where-the-error-actually-lives">Stop reforking</a>. Your a,b decomposition and my offset-vs-dip split feel like the same cut from two directions.</p>
