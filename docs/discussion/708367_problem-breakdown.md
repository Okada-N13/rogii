# Problem Breakdown

- 投稿者: Tabish Shah Mohsin
- 投稿日時: 2026-06-15 09:39:22.878000
- 投票数: 35
- コメント数: 28（取得数: 28）
- トピックID: `708367`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/708367](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/708367)

## 本文

<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F28339484%2Fa31b02201c72d30f1ea15bd543a1275b%2FProblem%20Diagram.png?generation=1781516015139168&alt=media" alt="Problem Diagram"></p>
<p><em>Original source: <a href="https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697418">zacchaeus</a></em></p>
<ul>
<li>Modern drilling is a remote-controlled, blindfolded 3D underground navigation. Modern rigs use steerable drill bits.</li>
<li>A "horizontal well" starts by drilling vertically, but then gets curved and is maintained horizontally into the most oil yielding layer. Keeping the TVT constant. (Z may change)</li>
<li>Hence an <strong>L-Profile.</strong>, also called as a <strong>Horizontal Well</strong>. <a href="https://www.youtube.com/watch?v=Xr_kSHJguTM&t=167s">Video on Profiles.</a></li>
<li>Data is such that the layers including the ground surface are exactly <strong>parallel surfaces</strong>.<ul>
<li>Almost parallel planes.</li></ul></li>
<li><code>Goal</code>: Maintaining TVT to be constant. (Geosteering)</li>
<li><code>Current Objective</code>: Making a model for predicting current TVT.</li>
<li>The Drill Bit has a <strong>Gamma Ray (GR)</strong> tool/sensor. Different rocks naturally emit different amounts of radiation (Gamma Rays).<ul>
<li>For example, "Shale" (which holds oil) emits high gamma rays, while "Limestone" emits low gamma rays.</li></ul></li>
<li>A type well csv, is basically a lookup where <code>GR</code> vs <code>TVT</code> is mapped for a place.<ul>
<li>As found later, many wells share the sub sequence of a common type well.</li></ul></li>
<li>Everything meassured from sea level is -ve.</li>
</ul>

## コメント

### コメント 1 — Georgy Mamarin

- 投稿日時: 2026-06-27 08:43:11.697000
- 投票数: 0
- コメントID: `3482433`

<p>Following up on the GR-rotation thread above (Shrey's catch that the sensor rotates, and MY0705's point that the FFT rotation frequency is a denoising lever). I tested whether the denoise actually helps downstream. It does, but only a little: on a datum-localization check (does the GR/typewell misfit put its global minimum at the true TVT?), a plain rolling-median low-pass on GR (a crude stand-in for a proper rotation notch) moves localization from ~80% to ~84%. So MY0705's noise-reduction point shows up in the numbers, not just in theory. I haven't run a real FFT notch; it might do a bit more, or that +4 points may be most of what's there. The GR/typewell + piecewise-dip framing is what the rest of this rests on.</p>

### コメント 2 — Shrey Gandhi

- 投稿日時: 2026-06-17 15:34:31.283000
- 投票数: 6
- コメントID: `3474176`

<p>And I think the GR sensor is rotating too.</p>

#### コメント 2.1 — Tabish Shah Mohsin

- 投稿日時: 2026-06-17 16:54:16.530000
- 投票数: 4
- コメントID: `3474223`

<p>I would really like to understand and verify this. What specific observations made you believe that it could be rotating?</p>
<p>Edit: You seem to be exactly right!
<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F28339484%2F50da8b780bb5994bb9ebca4e77c33b8a%2FFFT.png?generation=1781715723678588&alt=media" alt=""></p>

##### コメント 2.1.1 — MY0705

- 投稿日時: 2026-06-19 02:06:09.757000
- 投票数: 2
- コメントID: `3474890`

<p>Indeed, we can see that the values ​​are higher at low frequencies! If there is a layer with a high GR value below, and the upper layer is also high, the GR value will increase each time the GR rotates, and this will be reflected as the frequency of the sensor's rotation. This might possibly be useful for noise reduction in the GR.</p>

##### コメント 2.1.2 — Shrey Gandhi

- 投稿日時: 2026-06-19 02:49:56.837000
- 投票数: 0
- コメントID: `3474906`

<p><a href="https://www.kaggle.com/tabishmohsin">@tabishmohsin</a> This is a good plot. How did you produce it?</p>

##### コメント 2.1.3 — Tabish Shah Mohsin

- 投稿日時: 2026-06-19 08:22:51.930000
- 投票数: 1
- コメントID: `3475047`

<p>The exact code <a href="https://www.kaggle.com/shreygandhi">@shreygandhi</a> :</p>
<pre><code>import numpy as np
from scipy.fft import fft, fftfreq

valid_gr = hw['GR'].interpolate().values 

N = len(valid_gr)
spacing = 0.10

yf = fft(valid_gr)
xf = fftfreq(N, spacing)[:N//2]

plt.figure(figsize=(10, 4))
plt.plot(xf[10:], np.abs(yf[10:N//2])) 
plt.title("FFT of Gamma Ray Signal")
plt.xlabel("Frequency (Cycles per Foot)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
</code></pre>

#### コメント 2.2 — Simon Beck

- 投稿日時: 2026-06-18 23:32:46.780000
- 投票数: 0
- コメントID: `3474867`

<p><a href="https://www.kaggle.com/shreygandhi">@shreygandhi</a> this is very powerful insight. Thx.</p>

##### コメント 2.2.1 — Shrey Gandhi

- 投稿日時: 2026-06-19 02:44:08.707000
- 投票数: 0
- コメントID: `3474904`

<p>Is it? How did you use it? </p>

### コメント 3 — Tabish Shah Mohsin

- 投稿日時: 2026-06-17 16:44:06.510000
- 投票数: 3
- コメントID: `3474218`

<p>Simply Submitting the last available input TVT gives <a href="https://www.kaggle.com/code/titericz/last-value-baseline/notebook">15.883 on <strong>L</strong>eader <strong>B</strong>oard.</a></p>

### コメント 4 — Rahul Mishra

- 投稿日時: 2026-06-30 04:31:03.877000
- 投票数: -1
- コメントID: `3484137`

<p>It's helpful but if someone can share their analysis code for a deeper understanding, that'd be great. </p>

#### コメント 4.1 — Georgy Mamarin

- 投稿日時: 2026-06-30 12:11:53.513000
- 投票数: 0
- コメントID: `3484459`

<p>The thing that helped me most here was splitting the error into two parts before modeling anything. There's a recoverable part: a per-well offset (pinned at the heel) plus a piecewise dip you read off the typewell by matching GR — that's most wells, and it gets you a long way. Then there's an irreducible part: on a minority of wells the datum is bimodal (the rhythmic bedding lets the GR line up at two depths a bundle apart), and there the best you can do is hedge, not commit. Tabish's breakdown above — the typewell-as-lookup and the piecewise "parallel jagged lines" — is exactly where that recoverable/irreducible split falls out.</p>
<p>I worked it through end to end with the measurement cells left visible if you want to audit each step: <a href="https://www.kaggle.com/code/georgymamarin/stop-reforking-where-the-error-actually-lives">where the ROGII error is recoverable vs irreducible</a>.</p>

##### コメント 4.1.1 — Rahul Mishra

- 投稿日時: 2026-06-30 13:18:41.493000
- 投票数: 0
- コメントID: `3484500`

<p>Thanks George, I'll surely look into it and comment again incase I needed your sharp analysis.</p>

### コメント 5 — Tabish Shah Mohsin

- 投稿日時: 2026-06-16 15:36:54.163000
- 投票数: 3
- コメントID: `3473448`

<p>Data Inconsistencies:</p>
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

### コメント 6 — Tabish Shah Mohsin

- 投稿日時: 2026-06-15 09:40:01.813000
- 投票数: 3
- コメントID: `3472837`

<p>Provided:</p>
<ul>
<li>Type Well: A master map of gamma-ray signals vs TVT.</li>
<li><code>GR</code>- Gamma Ray : Gamma Ray signal values sensed from the drill bit.</li>
<li><code>WELLNAME</code> - Unique identifier for the well.</li>
<li><code>MD</code> - Measured Depth (ft): Total length of all the pipe currently in the hole (Curve's Perimeter).</li>
<li><code>X</code>, <code>Y</code>, <code>Z</code> - Treat them as standard 3D Cartesian coordinates (ft). (Global Reference)</li>
<li><code>ANCC</code>, <code>ASTNU</code>, <code>ASTNL</code>, <code>EGFDU</code>, <code>EGFDL</code>, <code>BUDA</code> - Predicted depth of different types of layers.<ul>
<li>For the XYZ at what depth is the depth of the directly above <code>ANCC</code>: Check the value of <code>ANCC</code> for that row.</li>
<li>Only in the training data.</li></ul></li>
<li><code>TVT_input</code> - Some initial values of TVT.</li>
</ul>
<p>To Predict:</p>
<ul>
<li><code>TVT</code>: True Vertical Thickness<ul>
<li>Vertical: Along Gravity.</li>
<li>A loosely used term in the oil industry. Doesn't mean <a href="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQx0VWc2QpBTAH6y6KrZlkB9v8hVtXKXfd29w&s">this.</a></li>
<li>Here, it means the depth of the drill bit from the ground surface.</li>
<li>Know that ground surface isn't perpendicular to Z.</li></ul></li>
</ul>
<p>Nothing to do with TVT but <a href="https://www.youtube.com/watch?v=ygIIC4XNAX4">a video on  horizontal well.</a></p>

### コメント 7 — CoreyJamesLevinson

- 投稿日時: 2026-06-20 20:03:39.753000
- 投票数: 1
- コメントID: `3476702`

<p>Can you help me understand? If we are given the well path X, Y, Z then why do we even need to predict TVT? Can't we just track it based on the difference of the Z-value from the last known Z-value and from that measure TVT? Or is it because the thickness of the target layer changes as we drill through it?</p>

#### コメント 7.1 — PatrickAIForFun

- 投稿日時: 2026-06-21 04:23:12.907000
- 投票数: 3
- コメントID: `3476868`

<p>TVT is somewhat of a misnomer here because it is not the "thickness" of a layer but rather the vertical distance to some geology layer border which is set as the reference. This competitions data assumes that then all layers beneath are parallel and constant thickness per well. Thus the goal is more or less to predict the geological layer strucutre/dip.</p>

### コメント 8 — Reda Mountassir

- 投稿日時: 2026-06-17 14:54:39.590000
- 投票数: 1
- コメントID: `3474153`

<p>Thanks for sharing this explanation, it really helped!</p>

#### コメント 8.1 — Tabish Shah Mohsin

- 投稿日時: 2026-06-17 16:42:28.077000
- 投票数: 0
- コメントID: `3474215`

<p>You're welcome, glad I could help.</p>

### コメント 9 — Evdilos_Ikaria

- 投稿日時: 2026-06-17 04:13:36.813000
- 投票数: 2
- コメントID: `3473783`

<p>You made perfectly clear what TVT means.!!</p>

#### コメント 9.1 — Tabish Shah Mohsin

- 投稿日時: 2026-06-17 16:43:12.937000
- 投票数: 0
- コメントID: `3474216`

<p>Happy to clarify!</p>

### コメント 10 — Tabish Shah Mohsin

- 投稿日時: 2026-06-15 10:00:24.467000
- 投票数: 1
- コメントID: `3472848`

<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F28339484%2F7161173957e5bc8583c72b49431a32ee%2Fclass_instances.png?generation=1781517461321744&alt=media" alt="Class Distribution">
Type wells are divided into distinct groups, all of which are subsequences derived from a single master sequence.</p>

#### コメント 10.1 — Andrey Chankin

- 投稿日時: 2026-06-17 21:09:55.030000
- 投票数: 1
- コメントID: `3474348`

<p>hi <a href="https://www.kaggle.com/tabishmohsin">@tabishmohsin</a> , could you explain how did you find it?</p>

##### コメント 10.1.1 — Tabish Shah Mohsin

- 投稿日時: 2026-06-18 05:05:32.563000
- 投票数: 4
- コメントID: `3474506`

<p>The ks are same for these wells. Meaning if you get |ASTNL| + TVT - |Z| for a hw csv, you can see it to have the same value for the expression for the entirety of that hw. This value repeats over these wells.</p>
<p>Also you can do sequence/signal matching in tw to obtain this. You can also consider that TVTs have different intervals in tw.</p>

##### コメント 10.1.2 — Andrey Chankin

- 投稿日時: 2026-06-18 20:29:58.867000
- 投票数: 2
- コメントID: `3474831`

<blockquote>
  <p>The ks are same for these wells. Meaning if you get |ASTNL| + TVT - |Z| for a hw csv, you can see it to have the same value for the expression for the entirety of that hw. This value repeats over these wells.</p>
  <p>Also you can do sequence/signal matching in tw to obtain this. You can also consider that TVTs have different intervals in tw.</p>
</blockquote>
<p>do you mind sharing the code?</p>

##### コメント 10.1.3 — Tabish Shah Mohsin

- 投稿日時: 2026-06-19 08:34:16.803000
- 投票数: 0
- コメントID: `3475050`

<p>The code might not be worth sharing, I used an LLM to produce it. I wrote that comment just to include it for those who are just starting and would like to read what I have been able to make out of the problem.
Moreover others have reported in their discussions that they have been able to reduce type wells in less than 72 classes.</p>
<p>Thank You for the comment!</p>

##### コメント 10.1.4 — Tabish Shah Mohsin

- 投稿日時: 2026-06-19 09:40:15.337000
- 投票数: 2
- コメントID: `3475071`

<p>It was also found that the png files share the name of the type wells and there are about <a href="https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/705210">57</a> such type wells.</p>

### コメント 11 — Tabish Shah Mohsin

- 投稿日時: 2026-06-16 15:28:05.557000
- 投票数: 0
- コメントID: `3473436`

<p><a href="https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698449#3455674">Host's confirmation for "pseudo-typewells."</a></p>
<p>You can find a bunch more confirmations <a href="https://www.kaggle.com/igorakuvaev/discussion">here.</a></p>

### コメント 12 — Unknown

- 投稿日時: 2026-06-23 11:59:19.787000
- 投票数: 0
- コメントID: `3479124`

_本文なし_
