# Does pattern matching solutions (no ML) work here?

- 投稿者: Andrey Chankin
- 投稿日時: 2026-07-06 14:32:51.450000
- 投票数: 6
- コメント数: 3（取得数: 3）
- トピックID: `721549`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/721549](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/721549)

## 本文

<p>Hello everyone!</p>
<p>For most of the time I ve been working trying to implement something similar to what I have seen in the explanation video - try to match horizontal GR against typewell GR.</p>
<p>Yet I could not achieve a sustainable result, sometimes results were fine, but mostly for wells with significant TVT moves <img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2453410%2Fa60359f7820ba5c051d7b4398eb312ba%2Frogii_1.jpg?generation=1783347872147344&alt=media" alt="result_1"></p>
<p>But it doesnt even overperform baseline since for many "flat" wells predictions tend to go up/down too much.</p>
<p>I could not yet figure out how to correctly match almost flat parts. I even built a little soft to match GR signals by hand, but sometimes if very unobvious why specific regions has been chosen.</p>
<p>Could any of you achieve good results with such approach? - because I have tested numerous ideas here, but they all failed</p>

## コメント

### コメント 1 — Shrey Gandhi

- 投稿日時: 2026-07-07 08:05:56.307000
- 投票数: 3
- コメントID: `3493129`

<p>It will work for a lot of wells, but there will be many wells with bad overlap GR's.</p>

#### コメント 1.1 — Andrey Chankin

- 投稿日時: 2026-07-07 15:40:37.907000
- 投票数: 0
- コメントID: `3493316`

<blockquote>
  <p>It will work for a lot of wells, but there will be many wells with bad overlap GR's.</p>
</blockquote>
<p>I observe it as well, many wells with high RMSE for constant predictions are predicted well, but at the same time many almost flat wells are being ruined with my approach, thus the overall quality remains bad </p>

### コメント 2 — Ochir Dorzhiev

- 投稿日時: 2026-07-07 08:09:08.033000
- 投票数: 2
- コメントID: `3493132`

<p>I got similar results: we can build a candidate path set where very good paths exist even for hard cases, using likpf, pf, beam, formation, etc. The best oracle over these candidates is around 4.5–6 ft, close to the top LB scores.</p>
<p>The hard part is selection: horizontal GR vs typewell GR produces a very ambiguous cost map. An oracle-like scorer trained with positive paths and sampled negatives might help reduce this ambiguity, but only if the negatives are realistic hard negatives. Otherwise, the model may just learn to separate synthetic artifacts from real paths instead of learning what makes a geological path correct.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F295205%2F2fe63d91bdc11e548910960abaee2297%2FScreenshot%202026-07-07%20at%2015.15.55.png?generation=1783411791265873&alt=media" alt=""></p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F295205%2F177cd1396187c82f33de074a3a126633%2FScreenshot%202026-07-07%20at%2015.15.44.png?generation=1783411798928842&alt=media" alt=""></p>
