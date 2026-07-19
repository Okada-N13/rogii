# Surface columns (ANCC, ASTNU, etc.) are in TVD (Z), NOT in TVT

- 投稿者: Nicolas Bridelance
- 投稿日時: 2026-05-18 14:37:17.928000
- 投票数: 13
- コメント数: 2（取得数: 2）
- トピックID: `701034`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/701034](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/701034)

## 本文

<p>Hi everyone,</p>
<p>I spent some time debugging a visualization issue that might trip up someone somewhere, so sharing it here.</p>
<h3>The issue</h3>
<p>The six geological surface columns in the horizontal well files:</p>
<p>ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA </p>
<p>are stored as <strong>negative TVD values</strong> (same unit as the <code>Z</code> column, typically ranging from <code>-9500</code> to <code>-7500</code> ft), <strong>not</strong> in TVT.</p>
<p>If you try to plot them directly against TVT (which ranges from ~<code>11000</code> to <code>12000</code> ft), they'll appear completely off-scale and useless.</p>
<h3>Quick fix</h3>
<p>Map each surface value from Z-space to TVT-space using the well's own Z→TVT relationship. Since Z and TVT are almost perfectly correlated (|r| ≈ 0.999), a simple linear interpolation works well:</p>
<pre><code>from scipy.interpolate import interp1d

hw_clean = hw.dropna(subset=['Z', 'TVT']).sort_values('Z')
z_to_tvt = interp1d(hw_clean['Z'].values, hw_clean['TVT'].values,
                    kind='linear', bounds_error=False, fill_value='extrapolate')

surface_tvt = float(z_to_tvt(hw['ANCC'].dropna().iloc[0]))
</code></pre>
<p>Why it matters for modeling
Using raw Z values as if they were TVT will introduce an error of ~20,000 ft (the offset between negative TVD and positive TVT).</p>
<p>Hope this saves someone a few hours! 🙂</p>

## コメント

### コメント 1 — shanzhong8

- 投稿日時: 2026-05-19 07:04:28.703000
- 投票数: 0
- コメントID: `3460850`

<p>Does this preprocessing strategy result in information leakage?</p>

#### コメント 1.1 — Nicolas Bridelance

- 投稿日時: 2026-05-21 18:34:59.290000
- 投票数: 0
- コメントID: `3461909`

<p>Good question. The mapping doesn't introduce leakage because:</p>
<p>Z (TVD) is fully provided for all wells including test — it's the borehole trajectory, independent of the TVT target
The interpolator uses Z ↔ TVT_input pairs from the known (non-masked) portion of the well
The geological surface depths (ANCC etc.) are formation tops in the vertical section, well within the range covered by TVT_input
The transformation is purely geometric (coordinate conversion), not predictive. The TVT values you're actually predicting are at intermediate MD positions along the lateral, not at the surface depths.</p>
