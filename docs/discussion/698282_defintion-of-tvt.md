# Defintion of tvt

- 投稿者: Moonimonster
- 投稿日時: 2026-05-09 06:29:46.065000
- 投票数: 21
- コメント数: 13（取得数: 13）
- トピックID: `698282`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698282](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698282)

## 本文

<p>Hope y'all enjoying this competition!</p>
<p>I was wondering what you all defined this 'tvt', the target.
It seams like it means just literal 'depth' as described in typewell.
but I don't really think it is true meaning of tvt.. I searched up on internet, and I found out it means thickness of 'something'., but I'm not really sure what 'something' is in this competition.</p>
<p>Pretty sure, it would mean something more than 'depth', cause.. what would be the meaning of finding tvt where we have z right? 
I'm still working on this issue.
Hope some of you can share ideas about it. Thank you!</p>

## コメント

### コメント 1 — Tom

- 投稿日時: 2026-05-17 02:46:29.920000
- 投票数: 25
- コメントID: `3458945`

<p>Here are some cheetsheets I made. Hope this can help you understand. </p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4310004%2F1d43062d4baecad36ff39704c868f1f1%2Ff55e3557-e345-4263-8de0-eadcad56d4b7.png?generation=1778985960300442&alt=media" alt=""></p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4310004%2F6693b872d7340d187710fecf95555b57%2F9ef506e8-414e-4f1f-87d4-036883191086.png?generation=1778985972519640&alt=media" alt=""></p>

#### コメント 1.1 — Moonimonster

- 投稿日時: 2026-05-19 06:40:58.090000
- 投票数: 0
- コメントID: `3460835`

<p>Thank you!!😀</p>

### コメント 2 — PatrickAIForFun

- 投稿日時: 2026-05-09 11:14:22.527000
- 投票数: 3
- コメントID: `3455416`

<p>Please correct me if I am wrong here.
Based on my EDA and other discussions it seems that TVT in this case is the vertical distance to a virtual/imaginery reference line. This aligns somewhat with the usual notion of it being the vertical thickness measured from the upper part of a geological layer. The position of thie virtual 0 TVT layer is not known to us and varies per well and location but it follows the general geological structure. This is furthermore supported by the fact that ANCC - Z = TVT + , where some offset is a constant per well -> meaning that the TVT reference perfectly follows the geological structure and that the thickness of the uppermost layer (between TVT=0 and ANCC) is a fixed constant per well.
My believe is, that TVT=0 is referencing the ground level which is tilted in the same way as the geological structure.</p>
<p>These are all just suspicions and I am no domain expert nor did I do extensive EDA -> please correct me if I am wrong and discuss your thoughts.</p>

#### コメント 2.1 — Igor Kuvaev

- 投稿日時: 2026-05-10 04:02:10.810000
- 投票数: 6
- コメントID: `3455675`

<p>PatrickAIForFun
you are correct - TVT is vertical distance to a virtual/imaginery reference line
TVT=0 is referencing the ground level - this may not be the case. 
Most importantly TVT in the lateral corresponds to certain TVT in the typewell (for given lateral-typewell pair).
Lateral XYZ are correct - so laterals have a true spatial representation, while typewell is a vertical definition of the geology (GR values)</p>

##### コメント 2.1.1 — Evdilos_Ikaria

- 投稿日時: 2026-05-10 15:34:29.527000
- 投票数: 0
- コメントID: `3455888`

<p>Is it possible to give us a small sketch of TVT , Z  in a horizontal drill
Thanks</p>

##### コメント 2.1.2 — PatrickAIForFun

- 投稿日時: 2026-05-11 19:24:51.663000
- 投票数: 0
- コメントID: `3456382`

<p>Dear <a href="https://www.kaggle.com/igorakuvaev">@igorakuvaev</a> 
Thank you for the info and confirmation - this helps a lot.
A quick follow up question:
Are the TVT scales aligned across lateral and typewells? I.e. would it also be possible to steer a well against another nearby well instead of the linked typewell and get the correct TVT values (e.g. in a multi-steering setting)?
My first EDA regarding this is a bit suspicious:</p>
<ul>
<li>Plotting the TVT level from the topsets of given typewells across the full site does seem to make sense and follow the general stratigraphic dip.</li>
<li>Calculating where the virtual TVT=0 would be in Z space by calculating TVT + Z for each position and plotting these seems to show some discrepancies / offsets between wells which are not using the same typewell.
(See attached two images for this)</li>
</ul>
<p>Is this as expected or is multi-well steering not really an option here due to mismatched offsets between the TVT scales?</p>
<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F12887254%2F8ff5a8080e9b82cd9a93a46580a5ca20%2FFigure%2012.png?generation=1778527442861121&alt=media" alt="">
<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F12887254%2Fa3d55d68cf2775f501e537c2c2d7f21d%2FFigure%2014.png?generation=1778527452559133&alt=media" alt=""></p>

##### コメント 2.1.3 — Igor Kuvaev

- 投稿日時: 2026-05-14 06:10:16.320000
- 投票数: 3
- コメントID: `3457680`

<p>TVT is not aligned between different wells, it is only aligned between well and typewell</p>

##### コメント 2.1.4 — Dmitry Stadnik

- 投稿日時: 2026-05-18 11:13:13.573000
- 投票数: 0
- コメントID: `3459851`

<p>are well and typewell different physical wells drilled close to each other?</p>

##### コメント 2.1.5 — Brian Lynch

- 投稿日時: 2026-05-24 17:59:57.370000
- 投票数: 0
- コメントID: `3462725`

<p>This conflicts with what I’ve found online, but of course as the host what you say should be taken as truth for the competition!<img src="https://wiki.aapg.org/images/f/fd/Conversion-of-well-log-data-to-subsurface-stratigraphic-and-structural-information_fig2.png" alt=""></p>

### コメント 3 — hengck23

- 投稿日時: 2026-05-12 03:08:39.847000
- 投票数: -1
- コメントID: `3456504`

<p>this is chatgpt interpretation (from <a href="https://github.com/rogii-com/Python-SDK/tree/examples/tvt_rop_heatmap">https://github.com/rogii-com/Python-SDK/tree/examples/tvt_rop_heatmap</a>)</p>
<pre><code>XYZ = real spatial trajectory
GR = real log along trajectory
TVT = interpreted coordinate tied to linked typewell/interpretation
</code></pre>

#### コメント 3.1 — Moonimonster

- 投稿日時: 2026-05-12 05:51:03.723000
- 投票数: 0
- コメントID: `3456539`

<p>Thank you for sharing! 😀</p>

### コメント 4 — PC Jimmmy

- 投稿日時: 2026-05-09 15:32:47.693000
- 投票数: 0
- コメントID: `3455519`

<p>Using one of the AI's to help with coding I realized that it was confused about the meaning of tvt also :)</p>
<p>A general meaning could be that it was the thickness of the geological layer.  That's the assumption that the AI was using to write some code for me that was not working.</p>
<p>For us, not's not the correct definition, but rather its the distance from current drill bit point up.</p>

#### コメント 4.1 — Moonimonster

- 投稿日時: 2026-05-11 23:44:43.967000
- 投票数: 0
- コメントID: `3456461`

<p>Thank you for sharing your idea!😀</p>
