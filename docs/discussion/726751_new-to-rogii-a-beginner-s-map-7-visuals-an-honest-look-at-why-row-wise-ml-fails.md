# New to ROGII? A beginner's map — 7 visuals + an honest look at why row-wise ML fails

- 投稿者: n0Rollback 🇩🇿🇫🇷
- 投稿日時: 2026-07-16 12:11:00.180000
- 投票数: 9
- コメント数: 2（取得数: 2）
- トピックID: `726751`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/726751](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/726751)

## 本文

<p>If you just opened this competition and the columns (<code>TVT</code>, <code>ANCC</code>, <code>EGFDU</code>, a separate <code>typewell</code>…) look cryptic, I wrote a beginner notebook to get you un-stuck fast:</p>
<p><strong>🛢️ ROGII Geosteering for Beginners | 7 Visuals</strong> → <a href="https://www.kaggle.com/code/n0rollback/rogii-geosteering-for-beginners-7-visuals">https://www.kaggle.com/code/n0rollback/rogii-geosteering-for-beginners-7-visuals</a></p>
<p><strong>What it covers</strong></p>
<ul>
<li>A plain-language explanation of the real task: a well is steered <em>horizontally</em> through a thin pay zone, and we predict the bit's vertical position (<strong>TVT</strong>) from the <strong>gamma-ray (GR)</strong> log by matching it against a vertical reference well (the <strong>type well</strong>). That's geosteering.</li>
<li><strong>The official cross-section image</strong> that ships with every training well (the <code><id>.png</code> files) — the single most useful artifact in the dataset, and almost nobody opens it. The notebook shows how to read its four panels, then rebuilds each one.</li>
<li>7 visuals that make the GR ↔ TVT intuition click.</li>
</ul>
<p><strong>The part I think is most useful — an honest negative result</strong>
The dumbest baseline, "hold the last known TVT flat across the horizontal" (anchor-hold), scores <strong>RMSE ≈ 16.1</strong> out-of-fold on train wells. I then threw a straightforward row-wise LightGBM (GR shape + trajectory features) at it… and it scored <strong>17.463 — worse</strong>.</p>
<p>The lesson: this is a <strong>sequential</strong> problem. At any single row, GR alone barely tells you whether the bit is 3 ft high or low — the signal lives in the <em>sequence</em> and the type-well alignment, not in independent rows. If you're starting out, measure against anchor-hold first and make sure you actually beat it before trusting a per-row model.</p>
<p>Full notebook (code + all 7 plots): <a href="https://www.kaggle.com/code/n0rollback/rogii-geosteering-for-beginners-7-visuals">https://www.kaggle.com/code/n0rollback/rogii-geosteering-for-beginners-7-visuals</a></p>
<p>Hope it saves someone a few hours. Questions welcome 🙏</p>

## コメント

### コメント 1 — victor

- 投稿日時: 2026-07-17 20:03:51.740000
- 投票数: 0
- コメントID: `3499743`

<p>beginner map appreciated. advanced map still needed for my ego</p>

#### コメント 1.1 — n0Rollback 🇩🇿🇫🇷

- 投稿日時: 2026-07-17 20:07:21.530000
- 投票数: 0
- コメントID: `3499746`

<p>Ha, love the honesty 😄 The "advanced map" is where it gets brutal the real gains live in the sequence (type-well alignment, state-space / HMM-style tracking), which is the last section of the notebook. Full disclosure: I actually tried the fancy stuff (exact HMM smoother + particle filter) this week and it thoroughly humbled my ego on the leaderboard 😅. If I crack something that reliably beats a plain baseline, you'll get the advanced map. Thanks for reading 🙏</p>
