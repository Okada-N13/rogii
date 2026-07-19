# Why naive XGBoost hits a wall here — and what the literature suggests instead

- 投稿者: Nicolas Bridelance
- 投稿日時: 2026-05-18 14:47:01.160000
- 投票数: 28
- コメント数: 0（取得数: 0）
- トピックID: `701041`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/701041](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/701041)

## 本文

<h1>📚 Algorithm Review: Predicting TVT from GR Logs — From Signal Processing to Foundation Models</h1>
<p>I spent a good chunk of time reviewing the literature for this competition before
writing a single line of code. This post is that review, structured as a progressive
reading guide — from the geological basics to state-of-the-art deep learning.</p>
<p>The philosophy here: understanding <em>why</em> an algorithm works on this data is more
valuable than blindly throwing models at it.</p>
<hr>
<h2>1. The Physical Problem (Don't Skip This)</h2>
<p>Before any algorithm, let's be clear on what we're actually trying to do.</p>
<p>When a horizontal well is drilled, the drill bit travels <em>along</em> geological layers
rather than cutting through them vertically. This creates a fundamental ambiguity:
at any given MD (Measured Depth — the cable length along the borehole), <em>where
exactly are we in the rock pile?</em></p>
<p>This is what TVT answers. In this competition specifically, TVT is not a classical
layer thickness — it's a <strong>relative geological depth index</strong>: how far vertically
are we from a reference surface that follows the local geological structure.</p>
<p>The three depth coordinates:</p>
<ul>
<li><strong>MD</strong>: length along the borehole trajectory. The only coordinate tools actually
measure.</li>
<li><strong>TVD</strong> = $\int_0^{MD} \cos(I)\, dMD$ — true vertical depth. Straightforward
geometry.</li>
<li><strong>TVT</strong>: corrects for both borehole geometry <em>and</em> formation dip. This is what
tells you "you're 15 ft above the Buda limestone" regardless of how the well
meanders.</li>
</ul>
<h3>The Gamma Ray as a GPS in the rock</h3>
<p>GR measures natural radioactivity (K, U, Th) in API units. The key physical fact:</p>
<ul>
<li><strong>Shales</strong> concentrate radioactive minerals during deposition → <strong>high GR (> 100 API)</strong></li>
<li><strong>Carbonates / clean sandstones</strong> are chemically inert → <strong>low GR (< 40 API)</strong></li>
</ul>
<p>This means each geological formation has a characteristic GR "fingerprint". In the
Texas Eagle Ford play targeted here, the stratigraphy reads like a barcode:</p>
<table>
<thead>
<tr>
<th>Formation</th>
<th>GR character</th>
<th>Role</th>
</tr>
</thead>
<tbody>
<tr>
<td>Austin Chalk</td>
<td>Low, stable</td>
<td>Roof marker</td>
</tr>
<tr>
<td>Eagle Ford Shale</td>
<td>High, noisy (organic + carbonate interbeds)</td>
<td>The reservoir</td>
</tr>
<tr>
<td>Buda Limestone</td>
<td>Sharp drop</td>
<td>Floor marker</td>
</tr>
</tbody>
</table>
<p>The <strong>typewell</strong> is a vertical reference well that crossed all these layers in one
clean pass. It provides the canonical GR barcode for the area.</p>
<p><strong>The core problem</strong>: the lateral well's GR log is a <em>deformed version</em> of this
barcode — stretched, compressed, shifted — because the well meanders along the
layers rather than cutting through them. Our job is to continuously match the
deformed barcode against the reference to read the stratigraphic position (TVT).</p>
<hr>
<h2>2. Dynamic Time Warping — The Right Tool for Deformed Signals</h2>
<h3>Why classic correlation fails first</h3>
<p>The naive approach is a simple cross-correlation: slide the lateral GR window over
the typewell GR and find the best match. This works for a <strong>rigid shift</strong> (the same
sequence, just offset). But real geology doesn't work that way:</p>
<ul>
<li>A limestone bed might be 5 ft thick near one well and 12 ft thick 2 km away
(lateral thickness variations due to compaction, paleotopography…)</li>
<li>The drill bit may have spent more time in one layer if the well undulates</li>
</ul>
<p>The lateral GR signal is <strong>non-linearly stretched</strong> relative to the typewell. Classic
correlation produces garbage alignments in these cases.</p>
<h3>DTW: elastic alignment</h3>
<p>DTW (Sakoe & Chiba, 1978) — originally designed for speech recognition — solves
exactly this. The insight: instead of finding a single shift, find the optimal
<em>path</em> through a 2D cost matrix that warps both signals onto each other.</p>
<p>$$D(i,j) = c(x_i, y_j) + \min{D(i-1,j),\; D(i,j-1),\; D(i-1,j-1)}$$</p>
<p>where $c(x_i, y_j) = |x_i - y_j|$ is the local cost (GR difference).
The path must start at $(1,1)$ and end at $(N,M)$, cannot go backwards (monotonicity
constraint), and is found in $O(N \times M)$ by dynamic programming.</p>
<blockquote>
  <p><strong>Key intuition</strong>: DTW lets one point in the lateral log "match" multiple points
  in the typewell (or vice versa). That's the elastic stretching. A thin layer in
  the lateral that corresponds to a thick layer in the typewell is handled naturally.</p>
</blockquote>
<h3>Critical constraints for geology</h3>
<p>Without constraints, DTW is prone to degenerate alignments: a single point matching
an entire section. Two classical fixes:</p>
<p><strong>Sakoe-Chiba band</strong> — only allow alignments within $w$ samples of the diagonal:
$$|i - j| \leq w$$
This reduces complexity from $O(N^2)$ to $O(w \cdot N)$ and prevents geologically
absurd matchings (e.g. aligning an 11,000 ft TVT layer with a 9,000 ft layer).</p>
<p><strong>Itakura parallelogram</strong> — constrains path slopes to $[1/2, 2]$, forbidding extreme
compression or expansion. Appropriate when the geology is structurally simple and
layer thickness variations are bounded.</p>
<h3>Where vanilla DTW breaks down</h3>
<ul>
<li><strong>Faults / missing rock</strong>: the monotonicity condition ($i_{k-1} \leq i_k$) means
DTW cannot model repeated stratigraphy (reverse faults) or missing sections. You
need external domain knowledge to detect and handle these.</li>
<li><strong>Error accumulation</strong>: without external anchor points, alignment errors drift in
multi-well correlation loops (the loop doesn't close to zero).</li>
<li><strong>Open boundary problem</strong>: a lateral well rarely starts at the exact top of the
typewell sequence. Use <strong>Subsequence DTW</strong> (free start/end conditions on one side)
to handle this.</li>
</ul>
<hr>
<h2>3. Signal Correlation Methods — From Rigid to Elastic</h2>
<p>A useful mental model: think of these three methods as progressively more flexible
alignment tools.</p>
<h3>3.1 Classical Cross-Correlation — "Does it look the same if I shift it?"</h3>
<p>$$(f \star g)[\tau] = \sum_t f[t]\, g[t + \tau]$$</p>
<p>Slides $g$ over $f$, measures overlap. Fast, simple, but <strong>amplitude-sensitive</strong>:
different LWD tools or borehole diameters shift absolute GR values systematically,
creating false alignment peaks. Use only as a very rough first pass.</p>
<h3>3.2 Normalized Cross-Correlation (NCC) — "Does it have the same shape?"</h3>
<p>$$NCC(\tau) = \frac{\sum_t (f[t]-\bar{f})(g[t+\tau]-\bar{g})}
{\sqrt{\sum_t(f[t]-\bar{f})^2 \cdot \sum_t(g[t+\tau]-\bar{g})^2}}$$</p>
<p>This is the Pearson correlation coefficient computed on a sliding window. Result is
always in $[-1, 1]$, <strong>completely amplitude-invariant</strong>: if the lateral GR tool reads
20% higher than the typewell tool due to borehole conditions, NCC is unaffected.</p>
<blockquote>
  <p><strong>Practical rule of thumb</strong>: a sliding NCC score > 0.85 on a 30 m window is a
  high-confidence stratigraphic <strong>anchor point</strong>. NCC won't tell you about stretching,
  but it tells you "this wiggle in the lateral GR is definitely this wiggle in the
  typewell, with high confidence".</p>
</blockquote>
<p>In this dataset, we computed NCC scores across 100 wells: the median peak NCC is
~0.81, with ~31% of wells showing at least one anchor point above 0.75.</p>
<h3>3.3 Phase Correlation — "What's the exact shift, even in noise?"</h3>
<p>$$\Phi = \mathcal{F}^{-1}!\left(\frac{F(\omega)\,G^<em>(\omega)}{|F(\omega)\,G^</em>(\omega)|}\right)$$</p>
<p>Operates entirely in the frequency domain. If $g$ is a shifted copy of $f$, the
result $\Phi$ is a <strong>Dirac impulse</strong> at the exact offset — not a smooth correlation
peak. This gives sub-sample precision and is robust to broadband noise
(LWD tool background noise cancels out during normalization).</p>
<p>The limitation: same as classical cross-correlation, assumes a rigid shift. Best
used to detect <strong>coarse offset and faults</strong> at macro scale, before handing off to
DTW for local elastic alignment.</p>
<hr>
<h2>4. Python Library Benchmark</h2>
<table>
<thead>
<tr>
<th></th>
<th><code>dtaidistance</code></th>
<th><code>fastdtw</code></th>
<th><code>tslearn</code></th>
</tr>
</thead>
<tbody>
<tr>
<td>Backend</td>
<td>C / Cython</td>
<td>Pure Python</td>
<td>Python/Cython</td>
</tr>
<tr>
<td>Complexity</td>
<td>$O(w \cdot N)$ with band</td>
<td><strong>$O(N)$ approx.</strong></td>
<td>$O(N \cdot M)$</td>
</tr>
<tr>
<td>Multivariate</td>
<td>Limited (1D focus)</td>
<td>Yes</td>
<td>Yes (MDTW)</td>
</tr>
<tr>
<td>Special feature</td>
<td>Fastest exact 1D</td>
<td>Coarse-to-fine, linear memory</td>
<td><strong>Soft-DTW</strong> (differentiable)</td>
</tr>
<tr>
<td>Best for</td>
<td>Production, single-channel</td>
<td>Very long series (> 10k pts)</td>
<td>DL loss functions</td>
</tr>
</tbody>
</table>
<p>The <strong>Soft-DTW</strong> in <code>tslearn</code> deserves a special mention. Standard DTW uses a
$\min$ operator which is not differentiable — you can't backpropagate through it.
Soft-DTW (Cuturi & Blondel, 2017) replaces $\min$ with a smooth approximation
$\min_\gamma$, making the alignment loss <strong>trainable end-to-end in a neural network</strong>.
This matters if you want to embed DTW directly in a loss function (e.g. to penalize
stratigraphic misalignment in a LSTM or Transformer).</p>
<hr>
<h2>5. Can Foundation Models Help? A Critical Look</h2>
<p>The obvious question after reading about DTW and NCC: <em>"Can we just throw a
pretrained time series model at this and let it figure it out?"</em></p>
<p>The answer is: <strong>maybe, but not the way you'd expect</strong>. Here's why.</p>
<h3>What Foundation Models for Time Series actually do</h3>
<p>Over the past 2 years, several large pretrained models have emerged that work on
time series the same way LLMs work on text: train on billions of diverse sequences,
then zero-shot forecast on new data.</p>
<p><strong>The critical translation for this problem</strong>: our independent variable is
<strong>depth (MD)</strong>, not time. We're doing spatial inference along a borehole, not
temporal forecasting. That changes how these models should be used.</p>
<p>Also — and this is the key insight — <strong>we're not predicting GR</strong>. We're estimating
a hidden state variable (TVT) from an observed signal (GR) given a reference (typewell).
That's not a forecasting task, it's more like a Bayesian state estimation problem.
This means these models should ideally be used as <strong>feature extractors</strong> (encoder
representations → regression head), not as direct forecasters.</p>
<table>
<thead>
<tr>
<th>Model</th>
<th>Architecture</th>
<th>Spatial covariates</th>
<th>High-freq. detail</th>
<th>ROGII fit</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>TimesFM 2.5</strong> (Google)</td>
<td>Decoder-only (Patch)</td>
<td>✅ XReg</td>
<td>High</td>
<td><strong>8.5/10</strong></td>
</tr>
<tr>
<td><strong>Chronos</strong> (Amazon)</td>
<td>T5 Encoder-Decoder</td>
<td>❌ univariate</td>
<td>Medium</td>
<td>4/10</td>
</tr>
<tr>
<td><strong>MOIRAI</strong> (Salesforce)</td>
<td>Masked Encoder</td>
<td>✅ any-variate</td>
<td>High</td>
<td>7/10</td>
</tr>
<tr>
<td><strong>PatchTST</strong></td>
<td>Encoder-only (Patch)</td>
<td>✅ channel-indep.</td>
<td>Very high</td>
<td><strong>9/10</strong></td>
</tr>
</tbody>
</table>
<h3>Why Patching matters for GR logs</h3>
<p>A single GR measurement at one depth point carries almost no geological information
on its own — it's dominated by tool noise and borehole effects. The geological signal
lives in the <strong>shape of the curve over 5–15 ft</strong>: a peak-trough-peak pattern
identifies a formation boundary, a flat plateau identifies a homogeneous rock mass.</p>
<p>Patch-based models (PatchTST, TimesFM) encode the input as a sequence of local
windows ("patches") rather than individual points — exactly matching the scale at
which geology expresses itself in a GR log. This is not accidental: it's the right
inductive bias for this type of data.</p>
<h3>Why Chronos scores low here</h3>
<p>Chronos tokenizes the time series by normalizing and <strong>quantizing</strong> values into a
discrete vocabulary (like converting a signal into "words"). This works beautifully
for macroscopic trends in economic or climate data.</p>
<p>But in a GR log, a 1 m thin carbonate interlayer — a potential key stratigraphic
marker — appears as a brief high-amplitude spike. Quantization smooths or misclassifies
this spike, potentially erasing the very feature that pins the stratigraphic
correlation.</p>
<h3>The XReg advantage (TimesFM)</h3>
<p>This competition is fundamentally spatial: the same TVT can correspond to very
different GR values depending on location (X, Y, Z), lateral thickness variations,
and structural dip. Igor Kuvaev (ROGII) confirmed in the forum that XYZ coordinates
carry critical spatial information that GR alone cannot resolve.</p>
<p>TimesFM's <strong>XReg</strong> feature allows passing exogenous regressors (X, Y, Z, inclination,
surface depths from nearby wells) alongside the GR sequence as conditioning inputs.
That's exactly the architecture we need.</p>
<hr>
<h2>6. A Few Things the Forum Made Clear</h2>
<p>A direct quote from Igor Kuvaev (ROGII organizer) paraphrased:</p>
<blockquote>
  <p><em>"XYZ of lateral wells = true spatial position. The typewell = vertical geological
definition (GR as a function of TVT). Connecting the two is the problem."</em></p>
</blockquote>
<p>Some practical observations from top participants and organizers:</p>
<ul>
<li><strong>Naive tabular XGBoost</strong> on [X, Y, Z, MD, GR] hits a glass ceiling around ~9.7
public LB. It ignores the physical trajectory of the well.</li>
<li><strong>Geological layers are spatially continuous</strong>. A test well missing surface depth
data can have it imputed from its 10 nearest geographical neighbors with very high
accuracy (R² > 0.99 in cross-validation on the training set).</li>
<li><strong>Sequential context matters</strong>: the 50 points just before the PS (Prediction Start)
contain the local slope, GR trend, and last known TVT — the highest-value features
for immediate prediction.</li>
</ul>
<hr>
<h2>Summary</h2>
<p>The problem calls for a layered approach matching each method to what it does best:</p>
<table>
<thead>
<tr>
<th>Method</th>
<th>Handles stretching</th>
<th>Amplitude-invariant</th>
<th>Best use</th>
</tr>
</thead>
<tbody>
<tr>
<td>Classical cross-corr</td>
<td>❌</td>
<td>❌</td>
<td>Initial rough alignment</td>
</tr>
<tr>
<td>NCC (sliding window)</td>
<td>❌</td>
<td>✅</td>
<td>Anchor point detection</td>
</tr>
<tr>
<td>Phase correlation</td>
<td>❌</td>
<td>✅</td>
<td>Coarse fault/offset detection</td>
</tr>
<tr>
<td>DTW (Sakoe-Chiba)</td>
<td>✅</td>
<td>❌</td>
<td>Local elastic alignment</td>
</tr>
<tr>
<td>PatchTST / TimesFM (encoder)</td>
<td>✅</td>
<td>✅</td>
<td>Deep feature extraction</td>
</tr>
</tbody>
</table>
<p>Happy to discuss any of this — particularly the DTW implementation details
(subsequence DTW and open boundaries are tricky to get right). 🙂</p>

## コメント

_コメントなし_
