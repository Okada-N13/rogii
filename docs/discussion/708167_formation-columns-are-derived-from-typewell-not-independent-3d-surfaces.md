# Formation Columns Are Derived from Typewell, Not Independent 3D Surfaces

- 投稿者: franticXu
- 投稿日時: 2026-06-14 04:10:22.188000
- 投票数: 25
- コメント数: 9（取得数: 9）
- トピックID: `708167`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/708167](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/708167)

## 本文

<h1>Formation Columns Are Derived from Typewell, Not Independent 3D Surfaces</h1>
<p>While exploring the data, I found that the 6 formation columns (ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA) in the horizontal well CSVs are <strong>not independently modeled 3D structural surfaces</strong>. They are derived from the <strong>typewell's Geology TVT intervals + the parallel formation assumption</strong>. Here's the evidence.</p>
<hr>
<h2>1. Layer Thickness Is Constant Along Each Well</h2>
<p>For each well, I computed the difference between adjacent formation columns:</p>
<pre><code>geo_d = [ANCC-ASTNU, ASTNU-ASTNL, ASTNL-EGFDU, EGFDU-EGFDL, EGFDL-BUDA]
</code></pre>
<p><strong>Result: layer thickness is nearly constant along every well's trajectory</strong>, with variation ≤ 0.01 ft — which is just rounding noise from the 2-decimal-place precision of the original data.</p>
<table>
<thead>
<tr>
<th>Layer Pair</th>
<th>Example Well Mean (ft)</th>
<th>Std Along Well (ft)</th>
</tr>
</thead>
<tbody>
<tr>
<td>ANCC → ASTNU</td>
<td>145.04</td>
<td>8.9e-13</td>
</tr>
<tr>
<td>ASTNU → ASTNL</td>
<td>88.96</td>
<td>8.7e-13</td>
</tr>
<tr>
<td>ASTNL → EGFDU</td>
<td>122.70</td>
<td>7.3e-13</td>
</tr>
<tr>
<td>EGFDU → EGFDL</td>
<td>51.70</td>
<td>8.7e-13</td>
</tr>
<tr>
<td>EGFDL → BUDA</td>
<td>109.71</td>
<td>8.5e-13</td>
</tr>
</tbody>
</table>
<p>This means all 6 formation surfaces share the same structural shape, offset by constant thicknesses — the <strong>parallel formation assumption</strong>.</p>
<p>=</p>
<h2><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F11631026%2F9b9e429bcae2a8f0e7b460b395e71e5c%2Ffig1_geo_d_along_well.png?generation=1781409964395942&alt=media" alt="geo_d along well"></h2>
<h2>2. hw Layer Thickness ≈ tw Layer Thickness (97.5% of Wells)</h2>
<p>I compared the typewell's Geology TVT intervals with the horizontal well's formation column differences:</p>
<pre><code>tw thickness = typewell top_TVT - bottom_TVT for each formation
hw thickness = mean(horizontal_well[col_a] - horizontal_well[col_b])
</code></pre>
<p><strong>Result: for 754 out of 773 wells (97.5%), the two are essentially identical</strong> (difference < 1 ft).</p>
<p>This is the key evidence: the formation columns are not independent spatial models — they are <strong>derived from the typewell's TVT boundaries</strong> using the parallel assumption.</p>
<h2><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F11631026%2Fbdb0097623093e8d903d8e9bf7c349fe%2Ffig2_tw_vs_hw_scatter.png?generation=1781409995155626&alt=media" alt="tw vs hw scatter"></h2>
<h2>3. 7 Anomalous Wells: Typewell ANCC Upper Boundary Is Truncated</h2>
<p>7 wells show large discrepancies (> 10 ft) specifically in the ANCC-ASTNU layer:</p>
<table>
<thead>
<tr>
<th>well_id</th>
<th>tw ANCC-ASTNU (ft)</th>
<th>hw ANCC-ASTNU (ft)</th>
<th>Diff (ft)</th>
</tr>
</thead>
<tbody>
<tr>
<td>fa16d114</td>
<td>129.0</td>
<td>243.8</td>
<td>+114.8</td>
</tr>
<tr>
<td>aee6393a</td>
<td>126.5</td>
<td>190.1</td>
<td>+63.6</td>
</tr>
<tr>
<td>5138a660</td>
<td>101.3</td>
<td>156.2</td>
<td>+55.0</td>
</tr>
<tr>
<td>d5df0ff5</td>
<td>102.0</td>
<td>146.3</td>
<td>+44.3</td>
</tr>
<tr>
<td>c1d046f4</td>
<td>107.5</td>
<td>134.4</td>
<td>+26.9</td>
</tr>
<tr>
<td>28f4eda9</td>
<td>223.5</td>
<td>243.8</td>
<td>+20.3</td>
</tr>
<tr>
<td>56b00794</td>
<td>128.9</td>
<td>134.4</td>
<td>+12.6</td>
</tr>
</tbody>
</table>
<p><strong>Key observations:</strong></p>
<ul>
<li>Only ANCC (the shallowest formation) is affected; ASTNU and below match perfectly</li>
<li>Wells in the same geological area share identical ASTNU-BUDA boundaries, but differ in ANCC upper boundary</li>
<li>Root cause: <strong>incomplete ANCC labeling in some typewells</strong> — the ANCC top surface is truncated</li>
</ul>
<p>For example, wells <code>fa16d114</code> and <code>28f4eda9</code> have identical ASTNU-BUDA boundaries, but different ANCC upper limits (10627 vs 10533 TVT), while the hw reports ANCC-ASTNU = 243.8 ft for both — the true regional thickness.</p>
<h2><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F11631026%2F1374709692f06b17abfeb42356cb5fd0%2Ffig3_tw_vs_hw_per_layer.png?generation=1781410036481943&alt=media" alt="Per-layer hw - tw difference"></h2>
<h2>4. What TVT Actually Means Here</h2>
<p>TVT in this dataset is <strong>not</strong> the traditional "True Vertical Thickness" of a single formation layer. Instead:</p>
<ul>
<li>It is a <strong>cumulative vertical distance</strong> from a reference point (similar to TVD)</li>
<li>At each step: <code>TVT[i] = TVT[i-1] + |ΔZ_vertical|</code></li>
<li>In the horizontal section: vertical component ≈ 0 → TVT barely changes</li>
<li>The parallel formation assumption is the prerequisite: only when layers are parallel does vertical distance correctly represent how much formation has been traversed</li>
</ul>
<hr>
<h2>5. Implications for Modeling</h2>
<ol>
<li><p><strong>Low information content</strong>: The 6 formation columns ≈ 1 base surface + 5 constant offsets. The effective degrees of freedom is <strong>1</strong>, not 6.</p></li>
<li><p><strong>Test wells lack formation columns</strong>: Since they are derived from typewell data, test wells don't have them. Spatial interpolation from neighboring training wells is needed.</p></li>
<li><p><strong>The real challenge</strong>: Cross-regional structural variation (different base surface shapes at different locations), not intra-well formation identification.</p></li>
<li><p><strong>Practical simplification</strong>: If you can predict one formation depth for a test well, the other 5 follow from regional thickness statistics.</p></li>
</ol>
<hr>
<h2>Summary</h2>
<table>
<thead>
<tr>
<th>Finding</th>
<th>Evidence</th>
</tr>
</thead>
<tbody>
<tr>
<td>Formation surfaces are parallel</td>
<td>Layer thickness std < 0.01 ft along each well</td>
</tr>
<tr>
<td>Formation columns derived from typewell</td>
<td>97.5% of wells: tw thickness = hw thickness</td>
</tr>
<tr>
<td>TVT = cumulative vertical distance</td>
<td>Horizontal section TVT ≈ constant (vertical component ≈ 0)</td>
</tr>
<tr>
<td>7 wells have truncated ANCC</td>
<td>Only ANCC-ASTNU affected; deeper layers match perfectly</td>
</tr>
<tr>
<td>6 columns = 1 degree of freedom</td>
<td>All surfaces share one base shape with constant offsets</td>
</tr>
</tbody>
</table>
<p>Hope this helps with your modeling strategy. Happy to discuss further!</p>

## コメント

### コメント 1 — Georgy Mamarin

- 投稿日時: 2026-06-24 01:39:18.590000
- 投票数: 0
- コメントID: `3479643`

<p>Your 1-DoF result quietly underpins how I framed the whole task — once the six columns collapse to one surface plus constant offsets, predicting TVT is just reading that surface along the lateral, and "get one depth, the rest follow" falls right out. I leaned on it (and credited you) in a limits writeup I just put up. The follow-on I keep hitting: that surface is recoverable down to the offset and the dominant dip, but a minority of wells go bimodal where the GR matches the typewell at two positions about a bundle apart — and that's where most of my residual error ends up. Thanks for laying the structure out so cleanly.</p>

### コメント 2 — hengck23

- 投稿日時: 2026-06-14 09:41:36.967000
- 投票数: 1
- コメントID: `3472464`

<p>I wonder is the data real or synthetic?</p>

#### コメント 2.1 — franticXu

- 投稿日時: 2026-06-15 05:47:15.707000
- 投票数: 0
- コメントID: `3472761`

<p>It's all true; you can conduct the experiment yourself—just subtract the TVT values of different layers in the HW to obtain the result.</p>

#### コメント 2.2 — Tabish Shah Mohsin

- 投稿日時: 2026-06-16 15:34:12.473000
- 投票数: 0
- コメントID: `3473443`

<p>It seems that they might have interpreted the layers including the ground surface to be exact parallel rugged surfaces, mirroring every contour and depression.
<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F28339484%2Fa31b02201c72d30f1ea15bd543a1275b%2FProblem%20Diagram.png?generation=1781516015139168&alt=media" alt=""></p>

### コメント 3 — Mattias Fagerlund

- 投稿日時: 2026-06-14 06:29:31.323000
- 投票数: 1
- コメントID: `3472401`

<p>Nice research. There are a also 7 wells that have ANCC entirely absent** (the column exists but is all-NaN / blank in the
CSV).</p>

#### コメント 3.1 — Tabish Shah Mohsin

- 投稿日時: 2026-06-16 15:36:33.233000
- 投票数: 3
- コメントID: `3473446`

<table>
<thead>
<tr>
<th>Well Names</th>
<th>Reason</th>
<th>Remark</th>
</tr>
</thead>
<tbody>
<tr>
<td>059c8f24, 14ab73fb, eba6605e</td>
<td>Δ TVT is abrupt</td>
<td>Single row in each TW</td>
</tr>
<tr>
<td>1b1eba53, d7eb0be8, 81bf5923, 4c2208f5, 03a935ae, 727a3a10, a8ed028a</td>
<td>Empty ANCC</td>
<td>Whole column in HW</td>
</tr>
<tr>
<td>9dfff011</td>
<td>Empty EGFDL</td>
<td>Whole Column in HW</td>
</tr>
<tr>
<td>03a935ae, 1b1eba53, 4c2208f5, 4f3eb9e9, 727a3a10, 78a4a386, 81bf5923, a8ed028a, d7eb0be8, 5d7198fd, 6e9ccd38</td>
<td>Missing ANCC</td>
<td>in TW</td>
</tr>
<tr>
<td>86454a6f</td>
<td>Missing ANCC, ASTNU</td>
<td>in TW</td>
</tr>
<tr>
<td>0bbf5e67, 2cee0cba, 353e5502, 4f4ac5ce, 5aef5c6c, 5eae34a8, 99529c45, a87433c9, d60430e6, d00e7eb9, d90aa14c</td>
<td>Missing BUDA</td>
<td>in TW</td>
</tr>
</tbody>
</table>

### コメント 4 — konwarsky

- 投稿日時: 2026-06-20 07:44:27.817000
- 投票数: 0
- コメントID: `3475930`

<p>Thanks for sharing this observation. Can we assume that all the wells (horizontal) in the train and test data would be from the same neighborhood (X-Y space)</p>

### コメント 5 — Alchemist

- 投稿日時: 2026-06-16 02:47:03.713000
- 投票数: 0
- コメントID: `3473206`

<p>Interesting. But is the layer data of any help, given that it's not accessible in the test set ? </p>

#### コメント 5.1 — franticXu

- 投稿日時: 2026-06-16 08:33:54.127000
- 投票数: 0
- コメントID: `3473300`

<p>The purpose of my analysis of this part is to understand the structure information of the data. Because if the stratum information is independent, then it should be predicted. But obviously it is not. It is obtained by translating tw along x, y, and z. The subsequent direction should be focused on this set of data: md, x, y, z + hw_tvt_, hw_gr + tw_tvt, tw_gr. Through tw_gr + tw_tvt, we can learn the cross-sectional stratification and thereby narrow the scope (this part should be included in the pattern matching module of the gr information). My English is not good. This is machine translation. Please forgive me.</p>
