# How are you using the lithology labels?

- 投稿者: yiyu0716
- 投稿日時: 2026-06-06 03:06:11.597000
- 投票数: 5
- コメント数: 1（取得数: 1）
- トピックID: `704771`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/704771](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/704771)

## 本文

<p>Hi everyone,</p>
<p>I have been trying to make use of the lithology / Geology labels, but so far I have not found a reliable way to turn them into a clear validation improvement.</p>
<p>They seem useful for understanding the typewell structure and the geological context, but in my experiments they have not yet worked well as direct features or matching signals.</p>
<p>Has anyone found a good way to use these labels? For example, are you using them for segmentation, filtering, alignment, post-processing, or only for visualization / interpretation?</p>
<p>Any suggestions would be appreciated.😀</p>

## コメント

### コメント 1 — Tucker Arrants

- 投稿日時: 2026-06-06 05:53:01.670000
- 投票数: 2
- コメントID: `3467406`

<p>The obvious thing to try is using them as an auxiliary training task, but so far it has provided zero benefit in my pipeline. </p>
