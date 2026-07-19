# Are Public Notebooks Overfitting to the LB?

- 投稿者: k256.dev
- 投稿日時: 2026-06-12 15:17:00.227000
- 投票数: 4
- コメント数: 6（取得数: 6）
- トピックID: `707915`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/707915](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/707915)

## 本文

<p>Regarding the public notebooks that achieve high scores, do you think they are overfitting to the LB? Personally, I think they are.</p>
<p>In the previous competition I participated in, many notebooks appeared that were only slightly modified versions of the best public notebook, changing things like the random seed or the ensemble weights. As a result, many users ended up experiencing a shake-down.</p>
<p>It looks to me like something similar may be happening in this competition as well.</p>
<p>Strangely, I have not seen any discussion about this yet, so I thought I would raise the question.</p>
<p>I would be happy to hear your thoughts.</p>
<p>thx.</p>

## コメント

### コメント 1 — Georgy Mamarin

- 投稿日時: 2026-06-27 08:39:48.027000
- 投票数: 0
- コメントID: `3482429`

<p>Late to this, but there's a number on it now. pilkwang put one on the seed band: six byte-identical notebooks of his scored 7.201–7.286 on the public board purely from particle-filter reseeding, nothing changed in the code (a wider near-identical set stretches to 7.168). So a ~0.03 gain over a parent could sit inside that band — worth checking whether it's a real method change or just a better seed before trusting it. The useful corollary for your silver/bronze point: you can self-check. Run your model under a few seeds and on a field-grouped hold-out; if the gain doesn't clear the seed band, it probably won't survive the private split. I have a small seed/hold-out check set up for exactly that if it's useful to anyone. This isn't about calling out anyone's score. It just lets you tell a real gain from seed luck before the reveal.</p>

### コメント 2 — hengck23

- 投稿日時: 2026-06-13 04:22:04.993000
- 投票数: 4
- コメントID: `3472016`

<p>note that some public notebooks may give different results if you submit multiple times due to "random seeding".  This is already a warning sign that public/private score may be different</p>

#### コメント 2.1 — ImperfectKitto

- 投稿日時: 2026-06-13 14:26:39.027000
- 投票数: 1
- コメントID: `3472195`

<p>That's true. One should fix seeds for experiments so noise doesn't get mistaken for genuine improvement </p>

### コメント 3 — ImperfectKitto

- 投稿日時: 2026-06-12 16:46:58.103000
- 投票数: 2
- コメントID: `3471855`

<p>For me, every LB improvement was prefaced by CV improvement. And I didn't rely on public notebooks much (or those public solutions blending ideas).</p>
<p>I would expect that's the case for other LB leaders, so I don't think there's much overfitting</p>

#### コメント 3.1 — k256.dev

- 投稿日時: 2026-06-12 17:35:10.503000
- 投票数: 2
- コメントID: `3471874`

<p>I can imagine that your score is not overfitted. I also agree that prioritizing CV is the most important thing.</p>
<p>I think the higher-ranking participants, especially those around the gold medal range, are less likely to be overfitting. That was also the case in the previous competition.</p>
<p>However, I suspect that many participants around the silver/bronze medal range, whose scores are close to those of the public notebooks, are likely overfitting.</p>

### コメント 4 — Unknown

- 投稿日時: 2026-06-23 14:33:23.607000
- 投票数: -3
- コメントID: `3479207`

_本文なし_
