# stage.1 : global search using linear prior tvt = linear(md,z)

- 投稿者: hengck23
- 投稿日時: 2026-05-13 15:49:38.319000
- 投票数: 27
- コメント数: 5（取得数: 5）
- トピックID: `699326`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699326](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699326)

## 本文

<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F1aabc18ee91163e6ba228c5e1db9b147%2FSelection_3505.png?generation=1778687376098048&alt=media" alt=""></p>
<p>Next post: stage.2 iterative local search for refinement </p>

## コメント

### コメント 1 — hengck23

- 投稿日時: 2026-05-14 03:21:23.183000
- 投票数: 6
- コメントID: `3457623`

<p>this is why prior (constraints) is important.</p>
<p>lower GR fitting doesn't mean lower T
** it is an inverse problem ! **</p>
<p>You should learn the 2-parameter prior space using TVT RMSE, not GR RMSE.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F862c8e7dcb7c2c0594d4c769da63c3ca%2FSelection_3518.png?generation=1778728817525025&alt=media" alt="">VT. </p>

### コメント 2 — hengck23

- 投稿日時: 2026-05-13 17:03:16.327000
- 投票数: 1
- コメントID: `3457377`

<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fc56ecec1fbf2e65293ed60b919a8d170%2FSelection_3507.png?generation=1778691758365065&alt=media" alt=""></p>
<p>initalise with fitted line of md,z after PS and also using tvt_input. need to think of  a way to make it "smooth"</p>

#### コメント 2.1 — hengck23

- 投稿日時: 2026-05-13 17:12:46.277000
- 投票数: 1
- コメントID: `3457380`

<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F56d68326c7bf5fb5347999c5889b0513%2FSelection_3508.png?generation=1778692190154785&alt=media" alt=""></p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F65ed9281146bfcc077582dd4d4e54495%2FSelection_3509.png?generation=1778692204746245&alt=media" alt=""></p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F528e2db812b1dec908fe6de6e42c9e74%2FSelection_3513.png?generation=1778693089162977&alt=media" alt=""></p>
<p>top to bottom: typewell TVT-GR after PS, typewell TVT-GS, horizontal MD-smoothedGS showing forward and reverse, horizontal MD-smoothedGS showing TVT as color, horizontal TVT-smoothedGS  </p>

##### コメント 2.1.1 — hengck23

- 投稿日時: 2026-05-13 17:21:44.083000
- 投票数: 3
- コメントID: `3457385`

<p>gemini suggested this but i haven't tried:
<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fdd4117e1392d8e76e50fec03ec6111df%2FSelection_3512.png?generation=1778692902303316&alt=media" alt="">
<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F3909dbc9cfac77444c5d3e6acfabc790%2FSelection_3511.png?generation=1778692880943488&alt=media" alt=""></p>

### コメント 3 — Franklin Gois

- 投稿日時: 2026-06-03 01:37:42.410000
- 投票数: 0
- コメントID: `3465896`

<p><a href="https://www.kaggle.com/hengck23">@hengck23</a> Thank you! </p>
