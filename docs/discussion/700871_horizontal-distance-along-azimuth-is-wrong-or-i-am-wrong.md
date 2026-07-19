# Horizontal distance along azimuth is wrong, or I am wrong?

- 投稿者: Dmitry Stadnik
- 投稿日時: 2026-05-18 09:37:55.943000
- 投票数: 6
- コメント数: 3（取得数: 3）
- トピックID: `700871`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/700871](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/700871)

## 本文

<p>I have 2 questions:</p>
<ol>
<li>Why is it in meters on png and pptx while every other length/depth is in ft?</li>
<li>I'm trying to calculate departure (horizontal distance, D) from MD (measured depth) and TD (true depth).
The formula I'm using is:
<code>D = sqrt(MD ^ 2 - TD ^ 2)</code>.
For the well 000d7d20 I see that D is in range 7000-14000 ft, while on png it's in range 0-5000 m (lower left chart). This is impossible: 5000 m is 16404.2 ft, which is very close to MDmax (total measured depth) of that well (16744 ft) - if that would be the case the well should go horizontal right from the surface.
Is my calculation incorrect?</li>
</ol>
<p>The first image is my chart, the 2nd is the one from provided png.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F460046%2Fd8a5625a1c6ad0baed50a030a2849d66%2FScreenshot%202026-05-18%20at%203.03.49PM.png?generation=1779098655913344&alt=media" alt="My chart"></p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F460046%2F16ce941dcee6702b7f25414d85964180%2FScreenshot%202026-05-18%20at%202.36.21PM.png?generation=1779097175711016&alt=media" alt="Chart from png"></p>

## コメント

### コメント 1 — Dmitry Stadnik

- 投稿日時: 2026-05-19 05:10:42.703000
- 投票数: 1
- コメントID: `3460812`

<p>My calculation was indeed wrong - the departure (horizontal projection of well's trajectory) must be calculated as cumulative sum of departures at each datapoint:</p>
<p><code>well_data['dMD'] = well_data['MD'].diff()</code></p>
<p><code>well_data['dZ'] = well_data['Z'].diff()</code></p>
<p><code>well_data['dDeparture'] = np.sqrt(well_data['dMD'] ** 2 - well_data['dZ'] ** 2)</code></p>
<p><code>well_data['departure'] = well_data['dDeparture'].cumsum()</code> </p>
<p>With that calculation I get proper number 5000 for the well, and I assume the units <code>m</code> were in png and pptx by mistake.</p>

### コメント 2 — konwarsky

- 投稿日時: 2026-06-13 12:58:58.443000
- 投票数: 0
- コメントID: `3472173`

<p>Could you please explain the azimuth angle calculation given the data.
I am unable to get the exact 86.5 Degree single number.</p>

### コメント 3 — Unknown

- 投稿日時: 2026-05-18 15:50:20.540000
- 投票数: 0
- コメントID: `3460073`

_本文なし_
