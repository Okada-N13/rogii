#   **"What's working beyond the public baseline for hidden test wells?"**

- 投稿者: Anjana mohan
- 投稿日時: 2026-07-06 17:28:26.686000
- 投票数: -2
- コメント数: 1（取得数: 1）
- トピックID: `722041`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/722041](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/722041)

## 本文

<p>Hi everyone,
My current best is <strong>8.012</strong> using:</p>
<ul>
<li>Contact override for visible test wells (RMSE ~0.01)</li>
<li>128-seed likelihood-weighted PF ensemble (4 scales: 3, 5, 8, 12)</li>
<li>14-config beam search ensemble</li>
<li>Adaptive per-well PF scale selection via holdout backtest (3 cut fracs)</li>
<li>Fleongg pretrained LGB models blend (45% SP45 + 55% fleongg)</li>
<li>Savitzky-Golay smoothing
Bronze cutoff is ~7.2. The gap is ~0.8 points which I believe is coming entirely from hidden test wells.
<strong>Specific questions:</strong></li>
</ul>
<ol>
<li>For hidden test wells, what's the single most impactful improvement beyond 128-seed lik-PF?</li>
<li>Is the Gold Overlay calibration helping on hidden wells or only on visible ones? I'm getting ~25 RMSE when I apply it to hidden wells.</li>
<li>Are people using the Bayesian forward-backward smoother? The CV returns NaN in my setup.</li>
<li>What <code>tau</code> and <code>w_pf</code> values are working best for post-processing on hidden wells?</li>
<li>Is there signal in resistivity or other columns beyond GR for hidden wells?</li>
</ol>
<h2>Any hints appreciated! 🙏</h2>

## コメント

### コメント 1 — H. Ashida

- 投稿日時: 2026-07-08 03:42:54.177000
- 投票数: 0
- コメントID: `3493578`

<p>Hi! A few quick pointers:</p>
<ol>
<li><p>Blend weight: You wrote 45% SP45 + 55% fleongg — the public-anchored setting is the reverse (0.55×SP45). We tried fleongg-heavier and it scored consistently worse on LB. Worth double-checking.</p></li>
<li><p>Gold overlay at ~25 RMSE = bug. It should make a small guarded move toward candidates (alpha-capped, clipped), never replace predictions. And align by MD interpolation, not row index — hidden rerun copies aren’t row-aligned.</p></li>
<li><p>FB smoother NaN: likelihood underflow — clamp with max(lik, 1e-300) before logs, skip very short prefixes.</p></li>
<li><p>w_pf ≈ 0 once the 128-seed lik-PF ensemble is already in the blend.</p></li>
<li><p>Hidden wells only have MD, X, Y, Z, GR, TVT_input — no resistivity, markers are stripped.</p></li>
</ol>
<p>Good luck! 🙏</p>
