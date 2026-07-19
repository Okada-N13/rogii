# Share an UI visualizer 

- 投稿者: Tom
- 投稿日時: 2026-05-17 12:47:20.845000
- 投票数: 91
- コメント数: 63（取得数: 63）
- トピックID: `700424`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/700424](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/700424)

## 本文

<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4310004%2F8fe8f43532b152aa5a7637b7d6f1a2fe%2FUI1.png?generation=1779022020724846&alt=media" alt=""></p>
<p>Feel free to use it: <a href="https://github.com/tom99763/rogii-viewer">https://github.com/tom99763/rogii-viewer</a>  </p>
<p>And some EDA & directions in attachments. I suggest people to read <code>glossary.html</code> first to clarify some confused definition. </p>
<p>I really like this explaination about TVT:</p>
<pre><code>Analogy: TVT is the “floor number” in a geological building

Imagine the geology as a completed high-rise building, where each floor represents a different rock layer (ANCC, Austin Chalk, Eagle Ford, Buda, etc.).

Typewell = the elevator shaft: it goes vertically from the top floor to the basement, recording “the GR at this layer is X” as it passes through each floor.
Horizontal well = a person walking inside the building:  moving along the hallway of a certain floor, sometimes going up or down between floors.
TVT = which floor is this person currently on?
</code></pre>
<h3>Update GR Mismatch Visualizer</h3>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4310004%2F5b91ffbfc5d669a76d10fe1cb409dcb5%2FGR.png?generation=1779030611267623&alt=media" alt=""></p>

## コメント

### コメント 1 — hengck23

- 投稿日時: 2026-06-03 23:52:18.897000
- 投票数: 5
- コメントID: `3466303`

<p>the public notebook (recent lb 8.63) uses meta-heuristics to decide k-beam or pf. this means some information is embedded at "whole well level", e.g. len of csv, min and max of well z, etc…</p>
<p>problem of PF is that it is local and sequential (error accumulation):<br>
p(s_t|s_t-1).</p>
<p>how about:
p(s_t|distribution up to s_t-1).  </p>
<p>or even 
p(s_t|distribution exluding to s_t).  </p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Feab42d972538dcdcf5912b11fc3086df%2FSelection_3869.png?generation=1780530617337253&alt=media" alt=""></p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F438338a874344cbae87002c86d67d751%2FSelection_3870.png?generation=1780530630436236&alt=media" alt=""></p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fc849d82db43ad30674f359a52e1e4e56%2FSelection_3871.png?generation=1780530643655204&alt=media" alt=""></p>
<p>reference:
Introducing DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation<br>
<a href="http://pub.sakana.ai/diffusionblocks">http://pub.sakana.ai/diffusionblocks</a>  </p>

### コメント 2 — Tom

- 投稿日時: 2026-06-03 12:58:36.763000
- 投票数: 5
- コメントID: `3466070`

<p>Explaination about current trick and physical meaning in attachment</p>
<p>And is it possible to do point sampling instead and making model predicting point connections. Just initial thoughts.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4310004%2F3dad47de922281e411d11a249991181c%2Fexample.png?generation=1780491961954126&alt=media" alt=""></p>

### コメント 3 — Tom

- 投稿日時: 2026-06-12 12:59:10.873000
- 投票数: 2
- コメントID: `3471783`

<p>Even though this was 5 years ago, I try seasonal trends again.</p>
<p><a href="https://www.kaggle.com/code/tom99763/tensorflow-probability-inference?kernelSessionId=79007085">https://www.kaggle.com/code/tom99763/tensorflow-probability-inference?kernelSessionId=79007085</a></p>

#### コメント 3.1 — Tom

- 投稿日時: 2026-06-12 15:12:53.600000
- 投票数: 0
- コメントID: `3471827`

<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4310004%2F2addfecdec246f6e27dc9016915f53a2%2F5255.png?generation=1781277165146138&alt=media" alt=""></p>

#### コメント 3.2 — hengck23

- 投稿日時: 2026-06-13 03:26:44.953000
- 投票数: 4
- コメントID: `3472009`

<p>i have a feeling that the interpretation is different for different wells (hence it is mixture mode). It is not a "optimization/prediction" problem but rather predicting the geologist preference for this particular well interpretation. My experiments show that per well model generally performs better but it is too slow to run as submission.</p>

#### コメント 3.3 — hengck23

- 投稿日時: 2026-06-13 20:24:15.563000
- 投票数: 3
- コメントID: `3472290`

<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fca0857b85060e729040d365bffc559a1%2FSelection_4046.png?generation=1781382155509617&alt=media" alt=""></p>
<p>if you are predicting trajectories, here is an alternative formulation. very much like onion layering of the Vesuvius Challenge - Surface Detection.</p>
<p>each value (or segment, e.g peak or valley) of the typewell GR is a trajectory. </p>

### コメント 4 — Tom

- 投稿日時: 2026-05-27 13:06:06.467000
- 投票数: 5
- コメントID: `3463617`

<p>Working on Neural SDE now. I discover that forward-stepping curriculum can make it start to learn.
This is really badass.
<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4310004%2Fd5b35c29393af30fe67920b379ef2185%2Ffield.png?generation=1779888583205959&alt=media" alt=""></p>

### コメント 5 — hengck23

- 投稿日時: 2026-05-18 20:11:02.220000
- 投票数: 5
- コメントID: `3460419`

<p><a href="https://youtu.be/VgzFt7xknGo?si=rwz9Kv2oi3ZwniBE">https://youtu.be/VgzFt7xknGo?si=rwz9Kv2oi3ZwniBE</a></p>
<p>time 27:50 shows (results? or the infrence  process?) ROGII automatic alignment and  segmentation</p>

### コメント 6 — Tom

- 投稿日時: 2026-05-23 05:48:45.080000
- 投票数: 3
- コメントID: `3462399`

<p>Share another approach: curvature integration with teacher forcing warm start</p>

### コメント 7 — Tom

- 投稿日時: 2026-05-23 01:45:41.363000
- 投票数: 3
- コメントID: `3462360`

<p>Sharing a diffeomorphic warping approach from my vesiuvius challenge solution. (Warp from a flat line instead</p>

#### コメント 7.1 — Tom

- 投稿日時: 2026-05-23 02:03:20.400000
- 投票数: 1
- コメントID: `3462362`

<p>So, building on Giba’s last-value <a href="https://www.kaggle.com/code/titericz/last-value-baseline">baseline</a>, a promising next step is to predict the direction at each MD position using sign(x), and then iteratively refine the correction magnitude over N steps.</p>

##### コメント 7.1.1 — Tom

- 投稿日時: 2026-05-23 02:08:10.620000
- 投票数: 2
- コメントID: `3462363`

<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4310004%2F166e648d61d3dfc05c97f85c5ff64b26%2F555.png?generation=1779502089436723&alt=media" alt=""></p>

##### コメント 7.1.2 — hengck23

- 投稿日時: 2026-05-23 03:18:42.403000
- 投票数: 2
- コメントID: `3462373`

<p>i have a slightly different but similar idea:</p>
<pre><code>treat it like a RL game:

input solution
while some stop condition not meet:
- analyse and select a segment
- push the segment up or down
- accept if  action produce better results, else reject
- repeat

such methods benefit from large data
</code></pre>
<p>for me, since CNN+SDF can give end-to-end solution and can fit train data, my next plan is:</p>
<ul>
<li>massive train data by sythetic generator</li>
<li>hierachy/iterative approcale to improve resoution </li>
</ul>
<hr>
<p>i treat normal and wraped  as a kind of TTA for esembling (and also augmentation for training) and different have different warping template</p>

##### コメント 7.1.3 — Tom

- 投稿日時: 2026-05-23 11:40:39.573000
- 投票数: 2
- コメントID: `3462455`

<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4310004%2Fbd55adc259858926aa5af71ed0c216b9%2F1.png?generation=1779536126303326&alt=media" alt=""></p>
<p>Now I develop a piecewise correction model by defining multiple pieces using two split points, t1 and t2, along md.</p>
<p>A piece is defined as:</p>
<p>$$
piece =
sign(correction_{t1}^{oof} - model_{t1}^{oof})
\times
sign(correction_{t2}^{oof} - model_{t2}^{oof})=-1
$$</p>
<p>The idea is to partition the space according to whether the correction directions are consistent between the two split points.</p>
<p>If the correction direction or magnitude inside a piece is incorrect, the model learns an adjustment term of the form:</p>
<p>$$
sign(\alpha)\cdot |\alpha|
$$</p>
<p>Here, sign(alpha) controls the correction direction, while |alpha| controls the correction magnitude.</p>
<p>This allows the model to learn localized residual corrections in a piecewise manner.</p>

### コメント 8 — Tom

- 投稿日時: 2026-05-23 14:01:53.093000
- 投票数: 4
- コメントID: `3462480`

<p>Probabilistic modeling is shining (No GBDT, finish in 20min)</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4310004%2F60e41418f8624915f2bca7b830f6bc98%2F1.png?generation=1779544886799314&alt=media" alt=""></p>

#### コメント 8.1 — Tucker Arrants

- 投稿日時: 2026-05-23 19:00:10.470000
- 投票数: 8
- コメントID: `3462542`

<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4379159%2F22175fc1da830b548c670f8312200156%2FScreenshot%202026-05-23%20145911.png?generation=1779562775107735&alt=media" alt=""></p>
<p>~1.2 ft behind you with simple UNet model. No physics constraints yet.</p>
<p>Pre-training on synthetic wells gave a decent boost…</p>

##### コメント 8.1.1 — Sangram Patil

- 投稿日時: 2026-06-06 17:22:58.513000
- 投票数: 2
- コメントID: `3467583`

<p>How are you using the 2D U-Net? My input is <code>[B, C, H, W]</code> and the output is <code>[B, H, W]</code>, but the model isn't performing well. I can't get the FT score below 14 no matter what I try. Do you have any suggestions?</p>

##### コメント 8.1.2 — Tucker Arrants

- 投稿日時: 2026-06-08 02:15:08.823000
- 投票数: 1
- コメントID: `3467931`

<p>Your output has the right shape. Ask yourself what each of the H rows along a single column is competing to be and whether you're scoring that competition, or just regressing its shadow.</p>
<p>Look at what you're not feeding it that you already have. Review what <a href="https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699853#3467823">Hengck is doing</a> for a concrete example.</p>

##### コメント 8.1.3 — hengck23

- 投稿日時: 2026-06-09 21:31:42.040000
- 投票数: 2
- コメントID: `3468828`

<p>"you're scoring that competition, or just regressing its shadow."  </p>
<p>within one column, SDF actually ranks all (typewell, horizontal) matches and gives results in "distance form". that is why it is so powerful.  </p>
<p>looks like regression, but we are actually doing ranking </p>

##### コメント 8.1.4 — Sangram Patil

- 投稿日時: 2026-06-10 05:00:43.733000
- 投票数: 2
- コメントID: `3468881`

<p>Thanks, guys, <a href="https://www.kaggle.com/tuckerarrants">@tuckerarrants</a> and <a href="https://www.kaggle.com/hengck23">@hengck23</a>. I'm still confused about most parts, so I've been using Claude and Gemini to help me understand them. At least I managed to build an SDF baseline that matches the CV-LB.</p>
<p><a href="https://www.kaggle.com/code/sangrampatil5150/lb-15-41-cnn-sdf-train">https://www.kaggle.com/code/sangrampatil5150/lb-15-41-cnn-sdf-train</a></p>
<p>If you have any suggestions for improving the current pipeline, it would be a huge help. Thanks!</p>

##### コメント 8.1.5 — hengck23

- 投稿日時: 2026-06-10 05:14:11.403000
- 投票数: 4
- コメントID: `3468882`

<p>Nice work getting an SDF baseline running.</p>
<p>One caution: “matches CV–LB” is useful but not a litmus test, but it is not enough to prove the implementation is correct.
e.g. A wrong local CV and a wrong LB submission can still have a similar gap.</p>
<p>I would suggest checking the pipeline in stages:</p>
<p>1) start with baseline submission = last tvt value: local CV =12, LB=15 (we now LB is about +2 worse than local)<br>
2) any learning model must perform better than local cv 12 if it is correct becuase if there is no better solution it sould predict null which leads to 12<br>
3) to check correct modeling:<br>
a) overfit train (this is upper bound if there is no generalisation error). SDF should easily gets to 8<br>
b) generalise validation to train. without augmentation, SDF should easily get 11+. you probably can reduce it to 9/10 with augmentation.<br>
c) submit to LB and try to keep gap. CV/LB consistent or less than 2  </p>

##### コメント 8.1.6 — Tucker Arrants

- 投稿日時: 2026-06-10 05:40:46.857000
- 投票数: 3
- コメントID: `3468885`

<p>Good start - keep trying and lean on the LLMs. It took me a little to get mine running, but it will give you a very good understanding of the problem / data once you do. I think there are a lot of approaches for this competition, so enjoy and be creative with your modeling.</p>

### コメント 9 — Tom

- 投稿日時: 2026-05-21 16:35:22.610000
- 投票数: 3
- コメントID: `3461852`

<p>about fourier formation perspective on this problem</p>

### コメント 10 — Tom

- 投稿日時: 2026-05-20 15:13:20.400000
- 投票数: 3
- コメントID: `3461432`

<p>On DTW inverse problem in the wavelet domain</p>

#### コメント 10.1 — hengck23

- 投稿日時: 2026-05-20 18:23:38.070000
- 投票数: 0
- コメントID: `3461519`

<p>in the typewell GR domain would be better?</p>

#### コメント 10.2 — Gaurav Rawat

- 投稿日時: 2026-05-23 15:10:54.597000
- 投票数: 0
- コメントID: `3462500`

<p>Did dtw work for you</p>

### コメント 11 — Tom

- 投稿日時: 2026-05-19 11:04:43.927000
- 投票数: 3
- コメントID: `3460928`

<p>Update: </p>
<p>Baysian Physical-informed SegFormer got very good result.  (0.94 cv score)</p>
<table>
<thead>
<tr>
<th>Fold</th>
<th>baseline (077o)</th>
<th>soft_seg input (077q)</th>
<th>Δ q-o</th>
<th>bpinn (077bpinn) ⭐⭐</th>
<th>Δ bpinn-q</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>9.45</td>
<td>9.18</td>
<td>−0.27</td>
<td>9.18</td>
<td>−0.00</td>
</tr>
<tr>
<td>2</td>
<td>10.73</td>
<td>10.43</td>
<td>−0.30</td>
<td>9.79</td>
<td>−0.64 ★</td>
</tr>
<tr>
<td>3</td>
<td>9.44</td>
<td>8.68</td>
<td>−0.76 ★</td>
<td>8.45</td>
<td>−0.23 ★</td>
</tr>
<tr>
<td>4</td>
<td>11.39</td>
<td>10.48</td>
<td>−0.91 ★</td>
<td>10.13</td>
<td>−0.35 ★</td>
</tr>
<tr>
<td>5</td>
<td>10.45</td>
<td>9.58</td>
<td>−0.87 ★</td>
<td>9.35</td>
<td>−0.23 ★</td>
</tr>
<tr>
<td>Overall</td>
<td>10.32</td>
<td>9.70</td>
<td>−0.63</td>
<td>9.40 ⭐⭐</td>
<td>−0.30</td>
</tr>
<tr>
<td>LB</td>
<td>10.576 ✓</td>
<td>TBD</td>
<td>–</td>
<td>TBD</td>
<td>–</td>
</tr>
</tbody>
</table>
<p>It can formulate measurement equations to constrain the neural network through probabilistic modeling. I believe this could push the limits of GBDTs.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4310004%2F1dd9a77e93f676bdb46343b65d9d8a53%2F213.png?generation=1779188800831447&alt=media" alt=""></p>

#### コメント 11.1 — hengck23

- 投稿日時: 2026-05-19 11:17:04.340000
- 投票数: 3
- コメントID: `3460930`

<p>If you consider just gr fitting aline, transformer is clearly better. Current notebook has better results because it is a fusion of multiple methods.</p>
<p>You should analyse 1. Comparison with last value baseline( and public plane fitting baseline) 2. Normalised rmse ( ie divided by std if signal) 3. Error coming from well that is not sharing common vertical well? There are only about 69 unique vertical wells.</p>
<p>If baseline is better than the worst cases, constraint to baseline should help </p>

##### コメント 11.1.1 — hengck23

- 投稿日時: 2026-05-19 13:11:43.413000
- 投票数: 1
- コメントID: `3460962`

<p>you can try to add the following to the transformer as features, it should improve  results by 1~2</p>
<ul>
<li>shared common typewell id</li>
<li>x,y,z, amz, inc</li>
<li>plane z  sampled from fitted geology plane  ancc, buda , etc ….</li>
<li>different sommon filter</li>
<li>drop segment in training</li>
</ul>
<p>in alphafold there is MSA  template matching to guide protein folding. in theory all neighbour wells  can be used as template and input to transformer </p>

##### コメント 11.1.2 — Simon Beck

- 投稿日時: 2026-06-21 08:23:30.123000
- 投票数: 0
- コメントID: `3477062`

<p><a href="https://www.kaggle.com/hengck23">@hengck23</a> now this is gold. i shoulde try it. thx</p>

#### コメント 11.2 — NobelK

- 投稿日時: 2026-05-21 18:08:32.407000
- 投票数: 0
- コメントID: `3461895`

<p>That's amazing!!
How is this learning being done?
I'm very curious about the details.</p>

### コメント 12 — hengck23

- 投稿日時: 2026-05-17 15:26:24.330000
- 投票数: 3
- コメントID: `3459270`

<p>U can add following to cnn feature:</p>
<ol>
<li>Self correlation, good for identifying moving reverse </li>
<li>Neighbouring well correlation </li>
</ol>

### コメント 13 — Tom

- 投稿日時: 2026-05-17 13:37:40.257000
- 投票数: 3
- コメントID: `3459198`

<p>This plot seems showing a hidden insight</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4310004%2Fd0df1bc4db7dfa9a0ac8a8689037f967%2Finsights.png?generation=1779025047785552&alt=media" alt=""></p>

### コメント 14 — Tom

- 投稿日時: 2026-05-18 09:11:14.423000
- 投票数: 4
- コメントID: `3459778`

<p>I made a NN-based approach with some probabilistic modeling similar to <a href="https://www.kaggle.com/jeroencottaar">@jeroencottaar</a> Yale/UNC-CH solution. I haven't submitted yet but it seems a good direction.</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4310004%2F501a4be8bddc4492da4fd2bfac3fbd52%2Fensemble_vs_truth_5wells_v2.png?generation=1779095472915796&alt=media" alt=""></p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4310004%2F44863581fda421aee8b3c431f68898ae%2Fheatmap_boxes_5wells.png?generation=1779096241756462&alt=media" alt=""></p>

#### コメント 14.1 — PatrickAIForFun

- 投稿日時: 2026-05-18 12:54:45.643000
- 投票数: 3
- コメントID: `3459916`

<p>When you say similar to Jeroen's solution, do you mean you actually modelled a prior and are optimizing the TVT math to minimize some measruement of mismatch between the TVTs? If yes, this is highly interesting, but I don't see how a CNN would fit into this.
What am I missing or are you willing to share more?</p>

##### コメント 14.1.1 — Tom

- 投稿日時: 2026-05-18 13:16:46.113000
- 投票数: 4
- コメントID: `3459940`

<p>Yes, I’ve actually built few priors and have been working on minimizing several measurements. I’ll release it once I have more completed experiments and the full map is built.</p>

##### コメント 14.1.2 — hengck23

- 投稿日時: 2026-05-18 13:23:58.057000
- 投票数: 3
- コメントID: `3459946`

<p>", but I don't see how a CNN would fit into this" . put your prior in the loss:  </p>
<p>loss = regression loss + classification loss  + "too different from the prior loss"  </p>
<p>if you use probability:<br>
too different from the prior loss = P( solution not drawn from prior disturibution)  </p>
<p>if you assume Gaussian, then it is related to "distance from prior solution"</p>
<hr>
<p>similarly, if you use physics or geology equations (as prior), then it is
loss = regression loss + classification loss  + "much much the solution follows physics equations"  </p>
<hr>
<p>if there is prior or physics, maybe predict residual is easier: solution = residual + prior </p>

##### コメント 14.1.3 — Tom

- 投稿日時: 2026-05-18 13:36:12.803000
- 投票数: 1
- コメントID: `3459956`

<p>I use this package: <a href="https://docs.pyro.ai/en/stable/">https://docs.pyro.ai/en/stable/</a></p>

### コメント 15 — hengck23

- 投稿日時: 2026-05-17 15:33:51.527000
- 投票数: 4
- コメントID: `3459276`

<p>Normal dtw assume monotonic seq and cannot match reverse index, so be careful if you use it.</p>

### コメント 16 — Tom

- 投稿日時: 2026-05-22 00:55:42.623000
- 投票数: 2
- コメントID: `3462031`

<p>single bpinn reach 9.+ lb</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4310004%2F5365423f114f60c6b5d6ed711665b09e%2F123432.png?generation=1779411315220089&alt=media" alt=""></p>

#### コメント 16.1 — Gaurav Rawat

- 投稿日時: 2026-05-22 03:25:17.843000
- 投票数: 1
- コメントID: `3462059`

<p>Super what CV you getting ?</p>

##### コメント 16.1.1 — Tom

- 投稿日時: 2026-05-22 03:36:27.337000
- 投票数: 0
- コメントID: `3462061`

<p>9.62 cv score</p>

##### コメント 16.1.2 — Gaurav Rawat

- 投稿日時: 2026-05-22 21:20:49.623000
- 投票数: 0
- コメントID: `3462322`

<p>ahh awesome cv GBDTs are bit worse ,,, </p>

### コメント 17 — Tom

- 投稿日時: 2026-05-19 07:53:52.450000
- 投票数: 2
- コメントID: `3460870`

<p>Soft input segment worked. Just reach 0.97 overall CV with SegFormer. </p>
<table>
<thead>
<tr>
<th>Fold</th>
<th>baseline</th>
<th>soft_seg input</th>
<th>Δ</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>9.45</td>
<td>9.18</td>
<td>−0.27</td>
</tr>
<tr>
<td>2</td>
<td>10.73</td>
<td>10.43</td>
<td>−0.30</td>
</tr>
<tr>
<td>3</td>
<td>9.44</td>
<td>8.68</td>
<td>−0.76 ★</td>
</tr>
<tr>
<td>4</td>
<td>11.39</td>
<td>10.48</td>
<td>−0.91 ★</td>
</tr>
<tr>
<td>5</td>
<td>10.45</td>
<td>9.58</td>
<td>−0.87 ★</td>
</tr>
<tr>
<td>Overall</td>
<td>10.32</td>
<td>9.70 ⭐</td>
<td>−0.63</td>
</tr>
<tr>
<td>LB</td>
<td>10.576</td>
<td>?</td>
<td></td>
</tr>
</tbody>
</table>

### コメント 18 — hengck23

- 投稿日時: 2026-05-19 06:32:27.897000
- 投票数: 2
- コメントID: `3460833`

<p>Azimuthal LWD Data Interpretation for UBCTDGeosteering Using a Physics-Informed Neural Network
<a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6000576">https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6000576</a></p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F0d83424de73277a9b0029db77e206b16%2FSelection_3682.png?generation=1779172309683194&alt=media" alt=""></p>
<p>"stay / steer_up / steer_down" this could be signed dip (-1,0,+1)</p>

#### コメント 18.1 — Tom

- 投稿日時: 2026-05-19 07:01:27.140000
- 投票数: 1
- コメントID: `3460849`

<p>I think there's a opportunity to add some physical constraint in each segments for special behavior. Like controlling the curvature. </p>

##### コメント 18.1.1 — Tom

- 投稿日時: 2026-05-19 07:30:47.063000
- 投票数: 2
- コメントID: `3460861`

<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4310004%2Ff7f2ffedb845a1bda8d8796e78a188d4%2Fensemble_vs_truth_5wells_v2.png?generation=1779175832490103&alt=media" alt=""></p>

##### コメント 18.1.2 — hengck23

- 投稿日時: 2026-05-20 00:44:48.217000
- 投票数: 1
- コメントID: `3461227`

<p>if you can set some equations on</p>
<pre><code>forward GR =   F.interpolate (predict_tvt, typewell_tvt, typewell_GR)

besides l2 loss, how to compare forward GR  and observation horizontal GR

or 

forward geological formation = func(TVT, ...)
</code></pre>

### コメント 19 — hengck23

- 投稿日時: 2026-05-18 14:06:14.800000
- 投票数: 2
- コメントID: `3459986`

<p>i have a suggestion for you. use seq transformer to learn segments (span of md of similar dip) and output as:</p>
<pre><code>segmentation output:
1111222222333333333333334444444444444444455666666666666666

auxilarly ouput
dip
xxxyyyzzzz

DTW
etc ....

tvt
</code></pre>
<p>actually segmenting should be the first step</p>
<p>you can measure the goodness of each segment by GR fitting or DTW, so it is a "physics model"</p>
<p>ask chatgpt to make segmentation ground truth by considering gradients (-1,0,+1) of tvt and link them up (or a very slow method called monotonic regression on gradient)</p>
<hr>
<p>note, the segmntation results is not unqiue. so we can have top-k segmentation prediction</p>

#### コメント 19.1 — Tom

- 投稿日時: 2026-05-19 05:47:19.473000
- 投票数: 4
- コメントID: `3460821`

<p>This is what a casual segformer can achieve for me now</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4310004%2F8cb928831c0531e60d76135b542d654b%2Fheatmap_boxes_5wells.png?generation=1779169616302808&alt=media" alt=""></p>

##### コメント 19.1.1 — Mohit

- 投稿日時: 2026-05-19 19:17:47.993000
- 投票数: 0
- コメントID: `3461107`

<p>even that much is good</p>

##### コメント 19.1.2 — Gaurav Rawat

- 投稿日時: 2026-05-20 02:00:11.063000
- 投票数: 0
- コメントID: `3461248`

<p>Thats great </p>

### コメント 20 — hengck23

- 投稿日時: 2026-05-18 12:01:20.630000
- 投票数: 2
- コメントID: `3459878`

<p>maybe this is useful for 
you: <a href="https://www.youtube.com/watch?v=fEf6i2A0jdo">https://www.youtube.com/watch?v=fEf6i2A0jdo</a><br>
<a href="https://www.youtube.com/watch?v=vQDbKR3NAlM">https://www.youtube.com/watch?v=vQDbKR3NAlM</a>  </p>
<p>check the lecture from 00 to 05 etc</p>

### コメント 21 — Gaurav Rawat

- 投稿日時: 2026-05-18 00:48:33.427000
- 投票数: 2
- コメントID: `3459554`

<p>love the eda via claude here very nice to understand the comp .. </p>

#### コメント 21.1 — Tom

- 投稿日時: 2026-05-18 01:51:56.097000
- 投票数: 3
- コメントID: `3459581`

<p>This comp is really complicated. There are many details haven't been coveraged</p>

##### コメント 21.1.1 — hengck23

- 投稿日時: 2026-05-18 05:09:43.720000
- 投票数: 3
- コメントID: `3459655`

<p>one of the few competitions left that humans must do the problem definition first before applying agent optimization</p>

##### コメント 21.1.2 — Tom

- 投稿日時: 2026-05-18 05:30:01.593000
- 投票数: 1
- コメントID: `3459666`

<p>Turning "plan mode" in Claude and carefully define the problem by myself worked well for me.</p>

##### コメント 21.1.3 — Tom

- 投稿日時: 2026-05-18 06:12:54.617000
- 投票数: 3
- コメントID: `3459684`

<p>Workflow (every new experiment):</p>
<ol>
<li>Restate the geological problem
(physical reality + constraints + signals)</li>
<li>Abstract it into a mathematical problem
(alignment? inpainting? inverse problem? assignment? consensus?)</li>
<li>Map each domain concept to a model component</li>
<li>Explore multiple reasonable formulations in parallel</li>
<li>Only implement after the formulation is finalized  </li>
</ol>

##### コメント 21.1.4 — Gaurav Rawat

- 投稿日時: 2026-05-18 14:21:38.300000
- 投票数: 0
- コメントID: `3459998`

<p>I try to do nowadays grill me for it to also grill before advising ..</p>

### コメント 22 — Durga Kumari

- 投稿日時: 2026-05-18 15:06:36.027000
- 投票数: -1
- コメントID: `3460033`

<p>This is incredibly helpful, especially the TVT analogy.</p>

### コメント 23 — Mohit

- 投稿日時: 2026-05-18 14:20:14.570000
- 投票数: -1
- コメントID: `3459997`

<p>Great work out there also how is nn aproach used here?</p>

### コメント 24 — Navneet

- 投稿日時: 2026-05-18 07:57:27.787000
- 投票数: -1
- コメントID: `3459736`

<p>Cool UI visualizer <a href="https://www.kaggle.com/tom99763">@tom99763</a> </p>

### コメント 25 — Unknown

- 投稿日時: 2026-05-17 19:12:14.420000
- 投票数: 0
- コメントID: `3459409`

_本文なし_
