# OOF vs LB: should we track worst-well improvements instead of only global RMSE?

- 投稿者: Zhenyu Zhang
- 投稿日時: 2026-05-17 09:20:41.450000
- 投票数: 6
- コメント数: 0（取得数: 0）
- トピックID: `700340`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/700340](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/700340)

## 本文

<p>I’m trying to understand how to evaluate real progress in this competition.</p>
<p>In our experiments, lower OOF RMSE does not always seem to imply a better LB score. That
  makes me wonder: how should we know whether a model change is actually better?</p>
<p>One idea is to look not only at the overall OOF, but also at the hardest wells. In our
  EDA, some consistently difficult wells are:</p>
<ul>
<li><p>86454a6f</p></li>
<li><p>fb03ae90</p></li>
<li><p>1b1eba53</p></li>
<li><p>389ae58f</p></li>
<li><p>896d15b9</p>
<p>These wells tend to have very large signed-bias errors across the whole well. So maybe a
model with slightly better OOF is not really better if it does not improve these failure
cases.</p>
<p>For model comparison, would you focus on overall OOF, fold consistency, worst-well
improvement, bias reduction, or another validation signal?</p>
<p>I’d be interested to hear how others judge whether an OOF gain is likely to transfer to
LB.</p></li>
</ul>

## コメント

_コメントなし_
