# Some question

- 投稿者: YePing Lin
- 投稿日時: 2026-05-05 18:18:55.055000
- 投票数: 7
- コメント数: 3（取得数: 3）
- トピックID: `697300`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697300](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697300)

## 本文

<p>Hello everyone, may I ask how to perform data augmentation in this kind of competition? Or rather, is data augmentation reasonable (because I'm not sure if it will work)? This is my first time participating in this kind of competition, so I don't know much about it. Thank you to all the experts for your guidance.</p>

## コメント

### コメント 1 — hengck23

- 投稿日時: 2026-05-06 12:01:51.690000
- 投票数: 2
- コメントID: `3453996`

<p>interpolate nearby wells? you have to imagine you are subsampling from a big 3d geological structure. augmentation here means creating more samples (possibly virtual ones) from this big structure</p>

### コメント 2 — Abdessamed Zetroni

- 投稿日時: 2026-05-05 19:25:21.987000
- 投票数: 1
- コメントID: `3453673`

<p>Data augmentation has limited use here because of the nature of the problem.</p>
<p>What could work: adding small noise to GR logs, or randomly shortening the known zone during training to simulate different eval lengths.</p>
<p>What probably won't help: flipping/rotating trajectories (wellbores have a physical direction) or standard image augmentation (this is sequential tabular data, not images).</p>
<p>The bigger insight for this competition is that TVT barely changes in horizontal wells since the drill is moving laterally through the same formation. Feature engineering around the trajectory and typewell reference will likely matter more than augmentation.</p>
<p>Good luck!</p>

#### コメント 2.1 — hengck23

- 投稿日時: 2026-05-06 12:06:47.123000
- 投票数: 4
- コメントID: `3453998`

<p>warp the GR data for both reference and input  … like linear elastic transform.</p>
<p>make aligned pairs reference and input to decide what kinds of noise, deformation to add</p>
