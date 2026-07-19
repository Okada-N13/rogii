# Write-UP: ROGII Wellbore Prediction: Anchors, GR Alignment, and Guarded Geosteering 

- 投稿者: FOYSAL
- 投稿日時: 2026-07-01 23:58:38.723000
- 投票数: 7
- コメント数: 3（取得数: 3）
- トピックID: `717445`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/717445](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/717445)

## 本文

<p>Hi everyone,</p>
<p>I have published my Working Note for the ROGII Wellbore Geology Prediction competition:</p>
<p>ROGII Wellbore Prediction: Anchors, GR Alignment, and Guarded Geosteering
<a href="https://www.kaggle.com/writeups/foysalemonshanto/rogii-wellbore-prediction-anchors-gr-alignment">https://www.kaggle.com/writeups/foysalemonshanto/rogii-wellbore-prediction-anchors-gr-alignment</a></p>
<p>The note focuses on the validation work behind the solution: why last-known TVT is such a strong anchor, where GR/typewell alignment helps, why naive slope extrapolation and ungated GR matching often fail, and how guarded contact/geosteering ideas shaped the final approach.</p>
<p>Feedback is very welcome. Thanks to the organizers and the community for the competition and discussions.</p>

## コメント

### コメント 1 — Georgy Mamarin

- 投稿日時: 2026-07-03 07:24:19.650000
- 投票数: -1
- コメントID: `3486935`

<p>Congrats on getting the note out. The guarded-geosteering framing is the part I'd single out — "where GR/typewell alignment helps" and where un-gated matching fails is a boundary I hit from the measurement side too, and it's oddly under-discussed given how much of the public score rides on it. Same for the last-known anchor being this strong: hard to beat, easy to under-respect.</p>
<p>One question: how did you choose the guard width around the anchor? From my side the real per-well excursions look like tens of feet, not hundreds, so anything that lets the matcher roam further mostly gives it room to be confidently wrong — I'm curious whether you sized the guard from the data, or tuned it, or reasoned from the geosteering side.</p>

#### コメント 1.1 — FOYSAL

- 投稿日時: 2026-07-03 08:43:07.490000
- 投票数: 0
- コメントID: `3486986`

<p>Thanks a lot, Georgy. I agree with your interpretation.</p>
<p>I did not treat the guard width as a free “let the matcher search anywhere” parameter. The main constraint came from the train-tail EDA: most hidden-tail TVT movement was only on the order of tens of feet, while the last-known <code>TVT_input</code> anchor was already very hard to beat. That made wide unconstrained GR matching dangerous, because it gave the matcher enough freedom to lock onto a visually plausible but wrong GR pattern.</p>
<p>So the guard was chosen from a combination of:</p>
<ol>
<li>train-tail movement statistics;</li>
<li>flat-anchor vs GR-match validation;</li>
<li>visible-prefix consistency checks;</li>
<li>candidate disagreement / movement caps.</li>
</ol>
<p>In practice, I treated the anchor as the default physical prior and allowed movement only when prefix evidence supported it. If a candidate required a large move away from the anchor, it needed much stronger visible-prefix support; otherwise it was clipped, downweighted, or rejected.</p>
<p>The most important lesson for me was that GR/typewell alignment is real, but the usable excursion range is usually small. Once the matcher is allowed to roam too far, it often becomes confidently wrong rather than more geologically correct.</p>

### コメント 2 — steubk

- 投稿日時: 2026-07-02 14:31:16.790000
- 投票数: 0
- コメントID: `3486439`

<p>don't forget to add writeup link in the proper thread <a href="https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/716699">https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/716699</a> !</p>
