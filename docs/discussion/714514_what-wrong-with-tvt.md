# What wrong with TVT

- 投稿者: Sangram Patil
- 投稿日時: 2026-06-26 12:56:52.763000
- 投票数: 21
- コメント数: 11（取得数: 11）
- トピックID: `714514`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/714514](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/714514)

## 本文

<p>How are the top players consistently achieving 6.5+ TVT scores? I still can't even reach 8.0. Whenever I achieve a decent validation score, the leaderboard shows a clear domain shift. Sometimes my model also overfits. I'm struggling to figure out what's actually wrong with the TVT target.</p>
<p>So far, I've tried these approaches:</p>
<p>lgb: [Lb: 13.]</p>
<p><a href="https://www.kaggle.com/code/sangrampatil5150/lb-13-741-lightgbm-only-24-features">https://www.kaggle.com/code/sangrampatil5150/lb-13-741-lightgbm-only-24-features</a></p>
<p>phy model: [Lb: 8.7]</p>
<p><a href="https://www.kaggle.com/code/sangrampatil5150/lb8-781-rogii-sel15-spread3">https://www.kaggle.com/code/sangrampatil5150/lb8-781-rogii-sel15-spread3</a></p>
<p>2DUnet [Lb: 15.]</p>
<p><a href="https://www.kaggle.com/code/sangrampatil5150/lb-15-481-cnn-sdf-train">https://www.kaggle.com/code/sangrampatil5150/lb-15-481-cnn-sdf-train</a></p>
<p>1dunet [Lb: 9.130] 1DUnet </p>
<p><a href="https://www.kaggle.com/code/sangrampatil5150/lb-9-130-1dunet">https://www.kaggle.com/code/sangrampatil5150/lb-9-130-1dunet</a></p>
<p>Imptur Formation Phy [LB: 27.0]</p>
<p><a href="https://www.kaggle.com/code/sangrampatil5150/fork-of-notebookd466c9dfe5">https://www.kaggle.com/code/sangrampatil5150/fork-of-notebookd466c9dfe5</a></p>
<p>Despite trying these different approaches, I'm still struggling to understand what I'm missing.</p>
<p>Could any of the top players share some insights into what made the biggest difference for them? I'm clearly missing something, and I'd really appreciate any guidance.</p>

## コメント

### コメント 1 — Ryo Takaki

- 投稿日時: 2026-06-27 11:34:39.913000
- 投票数: 16
- コメントID: `3482502`

<p>I’m sure my own LB score may also be overfit to the public LB to some extent, but I still wanted to share my thoughts in case they are helpful to the community.</p>
<p>I’m also trying to earn my first gold medal, so I don’t want to go into solution-specific details while the competition is still running. But I can at least share some general thoughts on the process.</p>
<p>When I first joined this competition, I also had many experiments where the CV looked promising but did not transfer to the LB at all. That was pretty frustrating. Over time, after running a large number of experiments every day, I gradually built up a better intuition, and the score started to improve little by little.</p>
<p>To be honest, even now I still don’t feel like I know the “correct” way to approach this competition. I’m still experimenting and making mistakes. But in the end, I think there is no shortcut: we have to think for ourselves, not blindly trust the analysis from AI agents, carefully interpret the experimental results, come up with the next hypothesis, and keep running that cycle.</p>
<p>One thing that helped me a lot was using the <a href="https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/700424">visualizer</a> shared by Tom. Looking at the actual well data and prediction results with my own eyes was very useful for understanding failure cases and getting ideas for the next experiments. I would strongly recommend using it if you are not already doing so.</p>
<p>I can’t share concrete solution ideas, but I just wanted to say that many of us are also struggling through trial and error. Let’s keep pushing for the remaining month or so.</p>

#### コメント 1.1 — Sangram Patil

- 投稿日時: 2026-06-27 15:58:16.537000
- 投票数: 0
- コメントID: `3482666`

<p>Thanks <a href="https://www.kaggle.com/ryotakaki">@ryotakaki</a> appreciated</p>

#### コメント 1.2 — Moonimonster

- 投稿日時: 2026-07-06 09:48:54.337000
- 投票数: 0
- コメントID: `3490461`

<p>thanks for sharing! hope you'll get your own first gold medal at the end!</p>

### コメント 2 — Tucker Arrants

- 投稿日時: 2026-06-26 18:07:09.177000
- 投票数: 12
- コメントID: `3482041`

<p>When you say your model overfits, how do you mean? If you mean it does not generalize well, that can be solved with augmentation and other regularization (weight decay, stochastic dropout, etc). Overfitting is actually <a href="https://karpathy.github.io/2019/04/25/recipe/">a good starting point</a> for training neural networks. </p>
<p>Also, if you have a CV->LB gap above ~3 feet, there is likely something wrong. If LB is 3 feet better than CV, you are probably overfitting the public LB like most of the shared GBDT notebooks. If LB is 3 feet <em>worse</em> than CV, you probably have leakage or an inference bug. (Public notebooks have both CV leakage <em>and</em> overfit the public LB so do not calibrate your expectations on them.) </p>
<p>In my experience, CV-LB gap shrinks as your CV gets lower, but the public LB is genuinely noisy due to how small the 
sample size is, so it is never going to be as stable as you'd like - you are locally validating on 773 wells and being scored on 25% of ~200 wells on public LB. </p>

#### コメント 2.1 — Sangram Patil

- 投稿日時: 2026-06-27 15:57:19.433000
- 投票数: 1
- コメントID: `3482664`

<p>Thanks <a href="https://www.kaggle.com/tuckerarrants">@tuckerarrants</a> </p>

#### コメント 2.2 — Andrey Chankin

- 投稿日時: 2026-06-28 18:24:13.653000
- 投票数: 0
- コメントID: `3483314`

<p>how do you know the quantity of test wells?</p>

##### コメント 2.2.1 — lightsource<3

- 投稿日時: 2026-06-28 18:30:51.953000
- 投票数: 2
- コメントID: `3483316`

<p><a href="https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/data">https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/data</a></p>
<blockquote>
  <p>test/ Contains the evaluation data for about 200 wells.</p>
</blockquote>

### コメント 3 — Georgy Mamarin

- 投稿日時: 2026-06-29 06:29:39.863000
- 投票数: 1
- コメントID: `3483531`

<p>Building on Tucker's CV-vs-LB point with one thing that helped me here. I spent a while convinced there was a magic feature I was missing too. What got me unstuck was splitting the error into a recoverable part and an irreducible one.</p>
<p>Recoverable: a per-well offset plus a slope read off the GR-to-typewell match. That's most of the gap from the ~16 ft flat baseline down toward the ~7 cluster, and it's where a tuned per-well model earns its keep. Irreducible: a heavy tail where ~10% of wells carry ~40% of the squared error, because the GR lines up with the typewell at two stratigraphic positions about one bundle apart, so the datum is a near coin-flip (souldrive's thread has the geology). On those wells, hedging between the two candidates beats committing to either.</p>
<p>The upside is that the split tells you where your headroom actually is. The tail isn't a feature you're missing, it's geology, so it's fine to leave it hedged and spend your energy on the offset and slope, which is the part that moves.</p>
<p>One check that helped me separate the two cases: before trusting a "gain," rerun the stochastic part of your pipeline a few times and see whether the delta clears the run-to-run band or just sits inside it (pilkwang put a clean number on that reseed spread in the comments). It's a few lines you can drop into your own pipeline. I also left a small version in a notebook that takes per-well OOF and prints your own oracle ceiling plus a wall test with a shuffled control, if a reference is useful, but the check itself is yours to reimplement. It won't solve a well, but it tells you which kind of error you're staring at.</p>

### コメント 4 — lucian kucera

- 投稿日時: 2026-06-27 15:46:10.820000
- 投票数: 0
- コメントID: `3482657`

<p>Idk i just looked at ur 1dunet [Lb: 9.130] 1DUnet and it looks very, good it imo has good potential.</p>
<p>I will try improve this architecture. </p>
<p>So far what i see u only have: 397941, which might be to little but idk.</p>
<p>Imo i can take this architecture to below 7.</p>
<p>I think 1d convolution will be the key in this competition. Also some form of unet.</p>

#### コメント 4.1 — Sangram Patil

- 投稿日時: 2026-06-27 15:56:48.827000
- 投票数: 0
- コメントID: `3482663`

<p>This approach has its limits, and I think I've already reached them. You can try adding more features, but unless you create features that capture everything or derive a formula that can accurately predict the formation column, it won't be very useful anymore. I don't think this approach can be pushed much further.</p>

##### コメント 4.1.1 — lucian kucera

- 投稿日時: 2026-06-27 15:57:42.517000
- 投票数: 0
- コメントID: `3482665`

<p>idk i will try to create some unet + conv1d aproach lets see how it goes, so far imo the most promissing.</p>
