# Curious how others are handling outlier wells in the spatial ANCC index

- 投稿者: Jordan
- 投稿日時: 2026-05-17 04:41:09.386000
- 投票数: 3
- コメント数: 0（取得数: 0）
- トピックID: `700285`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/700285](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/700285)

## 本文

<p>I'm using a centroid-level Kriging model (one point per well) to predict ANCC at test/training wells then feeding that into an ANCC-Z baseline. When I look at the spatial residuals, about 5-6% of wells have ANCC values that are significant outliers relative to their 10 nearest neighbors. Has anyone found that keeping or removing outliers has had meaningful differences?</p>
<p>I'm second guessing myself a bit here because if the wells reflect authentic geological observations and aren't data errors, it would likely make the spatial model worse in areas where it matters the most</p>

## コメント

_コメントなし_
