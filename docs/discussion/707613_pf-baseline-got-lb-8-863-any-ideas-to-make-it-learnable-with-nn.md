# PF baseline got LB 8.863 - any ideas to make it learnable with NN?

- 投稿者: NobelK
- 投稿日時: 2026-06-11 03:15:20.326000
- 投票数: 12
- コメント数: 8（取得数: 8）
- トピックID: `707613`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/707613](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/707613)

## 本文

<p>Hi everyone,</p>
<p>Based on the idea shared in this discussion:
<a href="https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/700424#3466070">https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/700424#3466070</a></p>
<p>I implemented a baseline using a Particle Filter (PF), and it achieved a public LB score of 8.863.</p>
<p>However, my current approach is still mostly rule-based / inference-only. It does not really learn from the training data. The PF works surprisingly well because it can sequentially track a plausible TVT trajectory using GR likelihood and physical constraints, but I am not sure how to move from this handcrafted PF approach to a learnable model.</p>
<p>My rough idea is to replace the PF with a Neural Network (NN), or at least make some part of the PF pipeline learnable. For example:</p>
<ul>
<li>train an NN to directly predict the TVT trajectory</li>
<li>train an NN to predict residual corrections on top of the PF output</li>
<li>train an NN to rank/select PF candidate trajectories</li>
<li>learn the transition / observation model used inside the PF</li>
<li>use something like a differentiable particle filter</li>
</ul>
<p>But I am not sure which formulation is the most practical for this competition. I tried asking LLMs for implementation ideas, but the answers were honestly too generic and not very actionable.</p>
<p>Does anyone have a smart idea, hint, or direction for turning this PF-style baseline into a trainable NN-based approach?</p>
<p>Also, if there is something important I am overlooking in this problem setup, I would really appreciate any comments.</p>
<p>Thanks!</p>

## コメント

### コメント 1 — Tom

- 投稿日時: 2026-06-11 06:36:19.363000
- 投票数: 6
- コメントID: `3471295`

<p>There are a lot of options you can do:</p>
<ul>
<li>sub sampling N particles, expand a window then train a network to do N-way classification. This way switching the problem from regression to block-wise inference. </li>
<li>recording particle vectors then predicting the offset to calibrate them, impacting the PF process. </li>
<li>For each particle step, train a generative models. </li>
</ul>

### コメント 2 — hengck23

- 投稿日時: 2026-06-11 04:10:39.210000
- 投票数: 4
- コメントID: `3471263`

<p>first you need to "measure" the performance of your current PF method:<br>
1) probability of generating  the "truth trajectory tvt" (or close to it)<br>
e.g. you always get at least  30  "truth trajectory tvt with rmse<10" for 100% of the validation data, given at least 500 particles.    </p>
<p>you always get at least  3  "truth trajectory tvt with rmse<4" for 90% of the validation data, given at least 500 particles.  </p>
<p>2) rank error of your scorer.<br>
how good your score correlates to tvt rmse? (gr rmse is not a good correlator of tvt rmse)</p>
<hr>
<p>then you can decide what to improve. here are the strategies:
1) improve generator probability
2) improve scorer ranking/correlation 
3) if you cannot improve your scorer, then improve your generator so that FP are not generated</p>

#### コメント 2.1 — hengck23

- 投稿日時: 2026-06-11 04:20:44.013000
- 投票数: 1
- コメントID: `3471265`

<p>" I tried asking LLMs for implementation ideas, but the answers were honestly too generic and not very actionable."</p>
<hr>
<p>hello chatgpt, here is my current PF performance:</p>
<ul>
<li>probability of generating the "truth trajectory tvt"<br><ul>
<li>30 "truth trajectory tvt with rmse<10" for 100% of the validation data, given at least 500 particles.  </li>
<li>3 "truth trajectory tvt with rmse<4" for 50% of the validation data, given at least 500 particles.  </li>
<li>also i find that only 50% follows the physics constraint under the equation : ….</li>
<li>i show you show plots of results </li></ul></li>
</ul>
<p>(1) i want to increase "rmse<4" for 80%, is it feasible?
(2) how to use  physics constraint for general improvement?</p>
<p>waht is the estimate of gain in LB if (1) is achieved. how to test the data if (1) is achivable? ….</p>

##### コメント 2.1.1 — NobelK

- 投稿日時: 2026-06-11 04:33:56.073000
- 投票数: 1
- コメントID: `3471267`

<p>Thank you so much for the very specific and helpful advice.</p>
<p>I will re-examine it. Thank you so much.</p>

#### コメント 2.2 — Shrey Gandhi

- 投稿日時: 2026-06-12 21:53:42.930000
- 投票数: 0
- コメントID: `3471967`

<p>Hey hengck, have u tried any strategy to deal with bad GR overlap wells?</p>

### コメント 3 — Georgy Mamarin

- 投稿日時: 2026-06-23 14:56:10.343000
- 投票数: -2
- コメントID: `3479226`

<p>On the NN angle — we went down it (CNN, GRU, GBM, even a typewell cross-attention seq2seq) and none beat a well-tuned multi-scale PF on an honest well-grouped CV; the learned models landed ~13-14 vs the PF's ~10. So the win, at least for us, was upstream of the model — measure the generator and scorer first, exactly like hengck23 said.</p>
<p>On the bad-GR-overlap wells Shrey asked about (most of the error for us too): the GR misfit at the true TVT is flat or even worse than a decoy one bundle away on the worst wells, so no legal test-time trigger separates "converged" from "stuck in a self-similar decoy." That's a ceiling on those wells, not a tuning bug — GR pulls ~16 down to ~10 but can't pin the fine slope where it overlaps.</p>
<p>I wrote the limits up with a fork-and-go harness for measuring a PF's ceiling (it's on my profile) — happy to run it on your OOF if you share it.</p>

### コメント 4 — Unknown

- 投稿日時: 2026-06-11 07:33:17.437000
- 投票数: -1
- コメントID: `3471307`

_本文なし_

#### コメント 4.1 — Unknown

- 投稿日時: 2026-06-11 07:34:30.347000
- 投票数: -1
- コメントID: `3471308`

_本文なし_
