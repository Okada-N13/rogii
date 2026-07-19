# Dynamic Programming for TVT Tracking: What Worked, What Didn't, and What the Gap Tells Us

- 投稿者: Matteo Niccoli
- 投稿日時: 2026-05-27 16:03:02.064000
- 投票数: 11
- コメント数: 3（取得数: 3）
- トピックID: `702919`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/702919](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/702919)

## 本文

<h1> </h1>
<p><strong>Dynamic Programming for TVT Tracking: What Worked, What Didn't, and What the Gap Tells Us</strong></p>
<hr>
<p>Following up on my <a href="https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/702131">earlier discussion on domain priors</a>, I tried a different approach to TVT tracking: dynamic programming (DP).</p>
<p><strong>The idea in plain terms:</strong> The bit traces a physical path through the formation. TVT at one position constrains what TVT can be at the next position; it can't jump 50 ft in one sample. Most trackers in this competition (particle filters, beam search) use heuristics to exploit this constraint. DP is the brute-force alternative: evaluate <em>every possible TVT path</em> through the typewell state space and pick the one that best matches the observed GR log, subject to a smoothness penalty. No randomness, no pruning, guaranteed global optimum for each parameter set. The trade-off is that you discretize the state space (~400 candidate TVT positions in a +/-200 ft window around the anchor).</p>
<p>I ran five configurations with different smoothness settings, from "stiff" (strongly resist TVT changes between positions, producing smooth paths that capture the broad trend) to "loose" (allow rapid TVT changes, producing responsive but noisier paths that track local dip). Ensembling these gives the model both structural trend and local sensitivity. These paths + their ensemble statistics + GR residuals at various offsets = 24-75 features fed into the same LightGBM.</p>
<p><strong>For the geophysicists:</strong> this is the same algorithm as Hale's dynamic warping of seismic images. Not conventional seed-based autotracking, but the DP-based approach that finds optimal shifts to align two signals. The cost functions are structurally identical:</p>
<p><em>Dynamic image warping (Hale, 2013) — find shifts u(t) that align image f to image g:</em></p>
<pre><code>min sum_i [ (f(t_i) - g(t_i + u_i))^2 + lambda * |u_i - u_{i-1}| ]
</code></pre>
<p><em>Geosteering TVT tracking (this notebook) — find TVT states that align lateral GR to typewell GR:</em></p>
<pre><code>min sum_i [ (GR_obs_i - GR_typewell[state_i])^2 / sigma + mu * |state_i - state_{i-1}| ]
</code></pre>
<p>Both minimize a data-fit term (how well the observed signal matches the reference at the candidate position) plus a regularization term (how much the shift or state jumps between adjacent positions). Replace "seismic trace" with "lateral GR log", "reference image" with "typewell", "shift" with "TVT state", and you have the same algorithm. The same DP structure appears in speech signal alignment (Sakoe and Chiba, 1978), sequence decoding (Viterbi, 1967), and seismic image warping (Hale, 2013). All are instances of Bellman's (1962) dynamic programming principle. Pure numpy, no Numba, deterministic, ~25 min for 773 wells.</p>
<p><strong>What worked:</strong></p>
<ul>
<li>The Viterbi ensemble mean became the #1 feature by LightGBM gain importance, ahead of all NCC and trajectory features</li>
<li>7 of the top 20 features are Viterbi-derived</li>
<li>OOF improved by 0.46 total (14.806 to 14.346) across two iterations (v1: +0.154 from 5 Viterbi paths, v2: +0.226 from expanded GR residual families + confidence features + 2-seed blend)</li>
</ul>
<p><strong>What didn't work:</strong></p>
<ul>
<li>LB barely moved: 14.081 vs 14.082 baseline (a one-thousandth of a foot improvement)</li>
<li>This is a textbook CV-overfitting signature: OOF improved 0.46 ft but LB improved 0.001 ft, while fold variance increased from 0.44 to 0.87. The Viterbi features fit the training distribution well but don't generalize to the held-out test wells</li>
<li>Post-processing (alpha/tau grid-searched on OOF) added only ~0.05</li>
<li>Confidence features (cost gap between best and second-best Viterbi path) contributed near-zero gain despite being theoretically the unique advantage of full-state DP over beam search</li>
<li>NCC-centered GR residual family was the least informative; anchor-centered was the most informative</li>
</ul>
<p><strong>The interesting tension:</strong> the model ranks Viterbi features #1 by importance but the score barely improves on LB. This means the Viterbi paths carry signal that is partially redundant with existing NCC features. Both measure GR-typewell similarity; the Viterbi adds sequential consistency but the observation model (point-by-point GR comparison) is too noisy for the consistency to help much.</p>
<p><strong>The lesson - it's the observation model, not the DP architecture:</strong></p>
<p>The Viterbi architecture is correct: it keeps all states alive (unlike beam search which prunes to 8-20) and guarantees the global optimum per parameter set. But its observation cost is crude. A single GR value is a weak discriminator between adjacent TVT positions. Independent validation from sleep3r's shuffled-GR experiment: their normal-GR top-10 heatmap coverage was essentially indistinguishable from shuffled GR (the shuffled version actually scored marginally higher, which is the noise floor). Point-by-point GR matching operates in the noise regime.</p>
<p>From reading the published top-scoring notebooks and discussion posts, the pattern I see is that the leading solutions compensate not with better decoders, but with a fundamentally different signal source: spatial structural information from neighboring wells. The ~5 ft gap between ~14 and ~9 RMSE appears to be not about more features or better inference over the GR-typewell match, but about adding a second, independent information channel.</p>
<p><strong>Quantifying the problem structure - two numbers that look contradictory but aren't:</strong></p>
<p>We tested the naive prediction TVT = -Z + C (flat formation assumption). In the lateral section: 55 ft RMSE. Per-well correlation between dTVT and -dZ increments: median r = 0.79 (r-squared = 0.62).</p>
<p>These are consistent, not contradictory. The r = 0.79 measures increment <em>direction</em> correlation: when Z goes down, TVT usually goes up, and vice versa. But the within-lateral dTVT/dZ slope is ~+0.057, not the -1 that the naive model assumes. Z wiggles ~14x more than TVT in the lateral because the bit is drilling roughly horizontal through a nearly flat formation. High directional correlation + near-zero slope + large cumulative drift coexist: the naive model accumulates a growing error despite "predicting the right direction" 62% of the time. Our GR matching reduces 55 to 14.3 ft by capturing the actual formation dip; the remaining gap to sub-10 requires spatial structure from neighboring wells.</p>
<p><strong>A cautionary tale - GR scaling:</strong></p>
<p>The first run produced completely flat paths (path_range = 0.0 for all five configs). Root cause: clean_gr() z-scores the GR, shrinking values to [-3, 3]. With z-scored GR, the observation cost was ~0.03/position while movement costs were 4-35. The DP never justified leaving the anchor. Fix: raw GR in API units for the Viterbi cost function, while NCC features continue using z-scored GR upstream. If you're building a physics-based cost function on top of normalized features, check the scaling.</p>
<p><strong>Cross-domain connections:</strong></p>
<p>The same DP structure appears across fields under different names:</p>
<ul>
<li>Seismic image warping: Hale (2013) - "dynamic warping"; Yan and Wu (2021) - "DP-based horizon extraction"</li>
<li>Speech recognition: Sakoe and Chiba (1978) - "dynamic time warping"</li>
<li>Communications: Viterbi (1967) - "Viterbi algorithm"</li>
<li>Bioinformatics: gene structure decoding - "HMM decoding"</li>
<li>Automated geosteering: Zeng, Bhaidasna, and Zou (IADC/SPE-230729-MS, 2026) - particle filter + DTW, validating the log-correlation approach in a commercial context</li>
</ul>
<p>The geoscience community knows this as "dynamic warping" or "DP-based horizon extraction"; the signal processing community calls it the "Viterbi algorithm." Four-phase structure is identical: initialize, forward-accumulate, minimize terminal state, backtrack.</p>
<p><strong>A second algorithmic pattern: structural guide + local matcher</strong></p>
<p>The DP discussion above covers the <em>sequential tracker</em> side of the problem. But from reviewing the top-scoring public notebooks in this competition (particularly the v43 spatial-pooling notebook and the inference-stack notebook), I see a second pattern at work: a smooth regional surface provides a structural backbone, and local signal matching corrects the residual.</p>
<p>In seismic interpretation, Gogia et al. (2020, Interpretation/SEG-dGB) built a hybrid horizon tracker that combines inversion-based dip flattening (a smooth surface fitted to the regional dip field) with similarity-based autotracking (local event correlation that snaps the surface to the correct reflector). Without the autotracker, the dip-only surface drifts off-phase; without the dip surface, the autotracker creates holes and loop-skips. The hybrid outperforms either component alone.</p>
<p>The same decomposition appears in this competition. The spatial structural backbone (formation plane fits from neighboring wells, implemented as FormationPlaneKNN in the top-scoring public notebooks) is the dip-flattening analog: a smooth regional surface that predicts TVT from spatial position alone. GR-typewell matching (NCC, Viterbi, particle filters) is the autotracker analog: it locks the prediction to the correct stratigraphic position using the local log signature. Neither alone reaches sub-10; the combination does.</p>
<p>This reframes the 10-vs-14 gap. Our notebook explored the local-matcher side thoroughly (three decoder variants, one observation model, one ceiling) but lacked the structural guide. The top-scoring notebooks I reviewed have both. The lesson generalizes: in any tracking problem where a smooth prior surface exists and local observations are noisy, the hybrid architecture (structural guide + local matcher) dominates either component.</p>
<p><strong>References:</strong></p>
<ul>
<li>Hale, D. (2013). Dynamic warping of seismic images. Geophysics, 78(2), S105-S115.</li>
<li>Yan, S. and Wu, X. (2021). Seismic horizon extraction with dynamic programming. Geophysics, 86(2), IM51-IM62.</li>
<li>Sakoe, H. and Chiba, S. (1978). Dynamic programming algorithm optimization for spoken word recognition. IEEE Trans. ASSP, 26, 43-49.</li>
<li>Gogia, R., Singh, R., de Groot, P., et al. (2020). Tracking 3D seismic horizons with a new hybrid tracking algorithm. Interpretation, 8(4), 1-7.</li>
<li>Zeng, Y., Bhaidasna, K., and Zou, A. (2026). IADC/SPE-230729-MS.</li>
</ul>
<p>Previous discussion on domain priors: <a href="https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/702131">link</a></p>

## コメント

### コメント 1 — Scott Weeden

- 投稿日時: 2026-07-14 20:45:03.707000
- 投票数: 0
- コメントID: `3496922`

<p>Have you tried a modified Dynamic programming verterbi approach using a modified at RFF approach.  I was able to achieve very good results using this research<br>
 <a href="https://gregorygundersen.com/blog/2019/12/23/random-fourier-features/">https://gregorygundersen.com/blog/2019/12/23/random-fourier-features/</a></p>

#### コメント 1.1 — Matteo Niccoli

- 投稿日時: 2026-07-16 12:45:13.967000
- 投票数: 0
- コメントID: `3499048`

<p>Thank you Scott
No, I hadn't tried RFF; to be honest it is the first time I hear about it, but it sounds like it could be used to attack the right part of the problem. What I found with this is that the bottleneck wasn't the path-finding step, it was the matching step underneath it: that is, how you score whether a stretch of lateral GR looks like a given depth in the typewell. I tried three alternative path-finders over the same matching score and they all came out the same. RFF would actually change the matching score itself, comparing the shape of a GR window rather than point-by-point differences, the part I'd expect to matter. Did you use it as the cost inside the DP, or as features feeding a model on top? I've wrapped up my work here, but curious how you set it up.</p>

##### コメント 1.1.1 — Scott Weeden

- 投稿日時: 2026-07-18 01:18:56.587000
- 投票数: 0
- コメントID: `3499816`

<p>It is now public:</p>
<ul>
<li>Dataset: <a href="https://www.kaggle.com/datasets/scottweeden/spatial-gp-hsmm-from-scratch">https://www.kaggle.com/datasets/scottweeden/spatial-gp-hsmm-from-scratch</a></li>
<li>Notebook: <a href="https://www.kaggle.com/code/scottweeden/rogii-gp-hsmm-viterbi-map">https://www.kaggle.com/code/scottweeden/rogii-gp-hsmm-viterbi-map</a>
uhm… hm now that you point it out, yes you hit the nail on the head its being used for weights.  I'm not sure what you mean by matching score though, It's a backwards propagation in my case. If you are trying to "backwards propagate" through a tunnel, it's the memory of what was chosen in the viterbi step. I'm not training all the wells at once I'm only doing 5 at a time. </li>
</ul>
