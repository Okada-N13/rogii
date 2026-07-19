# Any Advice on Cross-Validation Strategy?

- 投稿者: NobelK
- 投稿日時: 2026-05-18 06:48:51.684000
- 投票数: 4
- コメント数: 3（取得数: 3）
- トピックID: `700827`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/700827](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/700827)

## 本文

<p>I'm having a lot of difficulty coming up with a reliable cross-validation strategy for this competition.
If anyone has any helpful references, learning materials, or general advice on CV design, I would really appreciate it.
My intuition is that validation design is likely to be one of the most important keys to performing well in this competition.</p>

## コメント

### コメント 1 — PC Jimmmy

- 投稿日時: 2026-05-18 17:08:08.527000
- 投票数: 0
- コメントID: `3460182`

<p>The split I would recommend depends on your use of typewells.  A discussion <a href="https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698449">post</a> reported that a number of the typewells were duplicates.  </p>
<p>That analysis was partially correct (assuming I did not screw up).  </p>
<p>I have found that there are only 57 unique typewells in the full field.  If you called them Supertypes - than you would not want to have the same Supertype in both train and validation - again dependent on how your using typewells to build features.</p>

#### コメント 1.1 — hengck23

- 投稿日時: 2026-05-18 19:42:54.390000
- 投票数: 1
- コメントID: `3460374`

<p>Since you can see location and number of the hidden test wells from host explanation slides, create similar validation split based on that, mimicking the test distribution </p>

### コメント 2 — PatrickAIForFun

- 投稿日時: 2026-05-18 07:07:26
- 投票数: 0
- コメントID: `3459716`

<p>I would just go with standard Grouped K-Fold CV where the well id is the group. As per my understanding the hidden test set has wells very close to existing wells and also includes some typewells which are already present in the train set, thus there is no real value in trying to group the wells by location or duplicate typewells.
However, if your model is small enough / you can afford it: Leave-One-Well-Out would certainly be the best CV strategy.</p>
