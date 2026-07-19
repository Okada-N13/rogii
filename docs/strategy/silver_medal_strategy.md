# ROGII Wellbore Geology Prediction — 根拠ベース戦略 v2

> 初稿: 2026-07-19  
> 更新: 2026-07-19（Discussion、Working Note、公開 Notebook を再調査）  
> 最終提出期限: 2026-08-05

## 0. 結論

このコンペで最も筋の良い方針は、各行の TVT を直接回帰することではない。恒等式

```text
surface(MD) = TVT(MD) + Z(MD)
TVT(MD)     = surface(MD) - Z(MD)
```

を使い、既知で高周波な `-Z` の揺れはそのまま残し、well ごとに滑らかな `surface` の低周波トレンドだけを推定する。

主力は次の固定アンサンブルとする。

1. heel 適応した pseudo-typewell を使う multi-seed Particle Filter（PF）
2. PF の誤差を補正する小型の系列モデル（BiGRU または 1D CNN）
3. `surface` に対するロバストな一次・二次差分制約
4. PF と誤差相関が低い、小さな trellis/path posterior 成分

候補ごとに別モデルを選ぶ selector、近隣 well の直接転写、大型 Transformer は現時点の主戦略から外す。hard well の検出はできても、補正方向を当てる能力が確認できていないためである。

評価は well 単位 GroupKFold を基本にし、空間特徴を使う実験だけは leave-spatial-block-out も必須とする。差が 0.07 RMSE 未満なら、PF の再実行ノイズや fold 依存性を疑い、採用しない。

公開 sample test の 3 wells に対する same-well contact override は、train の真の TVT offset を使うため、通常のモデル性能とは分離する。これは core model の改善として扱わず、提出用の競技固有オプションとして隔離する。

## 1. 今回の調査で戦略を変えた点

| 旧方針 | 調査後の判断 |
|---|---|
| 公開 LB 6.021 以下を直接最適化 | private 期待値を主目的にし、公開 LB は参考値に限定 |
| PF 128×4 を 512 独立実行と解釈 | 公開コードは 128 seed の同じ軌跡を複数 likelihood 温度で再重み付けしている |
| selector で well ごとに最良候補を選ぶ | 候補 oracle は良くても selector は固定 blend より悪化。原則不採用 |
| 近隣 well、方位、空間特徴を積極利用 | 通常 GroupKFold では改善しても spatial-block CV で消失し得る。隔離実験に降格 |
| Transformer を有力な次段階とする | 22 種以上の NN、Transformer 系の否定結果が多い。小型 CNN/BiGRU を先行 |
| 同一 well override は「悪化しない」 | sample test が train のコピーで、真値 offset を使用。core model から除外 |
| 複数の補正を順に追加 | 3 個目の補正で大幅悪化した実例あり。常に完成スタック全体で再評価 |
| 小さな CV 改善も積み上げる | byte-identical PF でも公開値が約 0.04 揺れた。0.07 未満は原則ノイズ扱い |

## 2. 根拠の強さ

数値や主張を次の 3 段階で扱う。

- **A: 再現可能性が高い** — 公開コードで処理を確認できたもの、主催者の明示、複数の独立資料で一致したもの
- **B: 測定報告** — Working Note や Discussion に具体的な実験条件と結果があるが、このリポジトリでは未再現のもの
- **C: 未検証** — コメント、順位報告、非公開モデルの性能主張。仮説生成にだけ使う

重要な根拠は以下の通り。

| 根拠 | レベル | 戦略への反映 |
|---|---:|---|
| `TVT = surface - Z` で、推定対象は滑らかな surface trend | A | 全モデルの target/reconstruction を変更 |
| 誤差は一部の well に集中し、well 平均 bias が大きい | B | pooled RMSE と tail 指標を併記 |
| PF の seed ensemble と弱い GR likelihood が強い | A/B | 主力 baseline とする |
| heel local profile と typewell の中間 blend が全置換より良い | B | heel-adapted pseudo-typewell を採用 |
| fixed ensemble が learned selector より安定 | B | selector を原則使わない |
| spatial feature の改善が spatial-block で消えた | B | 空間特徴に追加 gate |
| NN では CNN/anchor が Transformer より安定したという報告 | B | 小型 residual model を優先 |
| per-well-only 完全系列モデルで pooled CV < 5 というコメント | C | 目標値や設計根拠には使わない |
| 近隣 profile copy が有効というコメント | C | 独立の空間実験としてのみ検証 |

## 3. 問題の捉え方

### 3.1 予測対象

trajectory の `Z` は test でも既知である。TVT の細かな wiggle の大部分は `-Z` から自動的に復元できるため、モデルには低周波な `surface = TVT + Z` を学習させる。

予測区間の直前に真の TVT が与えられる場合、最後の既知点を

```text
surface_anchor = TVT_last_known + Z_last_known
```

として使う。系列モデルの target は絶対 surface より、次のどちらかを優先する。

```text
delta_surface(MD) = surface(MD) - surface_anchor
surface_rate(MD)  = d surface / d MD
```

これにより well 固有 datum を直接学習する必要が減る。

### 3.2 難しさ

GR の自己相似性により、15–25 ft 程度ずれた複数の alignment が同程度に見える well がある。さらに大きな誤差は少数の well に集中する。Working Note では worst 5% wells が SSE の 52.5%、worst 10% が 64.5% を占め、well 平均誤差を oracle で除くと RMSE が約 5.01 まで下がったと報告されている。

したがって課題は、通常 well の局所的な wiggle を高精度化することより、hard well で datum・傾斜・分岐を誤る頻度を減らすことである。

## 4. 検証設計

### 4.1 基本 split

- 5-fold GroupKFold、group は `well_id`
- 同じ well の人工 cut を複数作る場合も、元 well と同じ fold に固定
- 特徴生成、pseudo-typewell、正規化、モデル学習は fold 内 train のみで fit
- test と同じく、cut より後の GR・Z・MD は見えるが TVT は隠す
- 重複行、同一 well ID、近接座標の混入を fold ごとに監査

### 4.2 必須指標

主指標は全行 pooled RMSE とする。ただし次も必ず記録する。

- well ごとの RMSE の median、P90、worst decile
- worst 5% / 10% wells が占める SSE
- well ごとの平均 bias、傾斜誤差
- 各 fold と各 seed の値
- baseline に対する well 単位 paired bootstrap の信頼区間

### 4.3 空間特徴の追加 gate

座標、近隣 well、field trend、方位を使うモデルは、通常 GroupKFold に加えて座標を K-means 等で 5–6 地域に分けた leave-spatial-block-out を通す。

採用条件は以下すべて。

1. 通常 GroupKFold で改善
2. spatial-block の平均でも改善
3. 過半数ではなく、原則すべての block で悪化しない
4. block 境界 well だけに改善が偏らない
5. 座標や近隣特徴を shuffle すると改善が消える

### 4.4 control-first の採用判定

各実験に次を付ける。

- **positive control:** 既知の PF baseline を再現できること
- **no-op control:** 重み 0、同一候補の重複 blend で値が変わらないこと
- **shuffle control:** 追加証拠を well/block 単位で shuffle すると改善が消えること
- **seed rerun:** 同設定で 3 回以上実行し、差が seed ノイズを上回ること
- **full-stack evaluation:** 単体でなく、最終 blend に追加した状態で評価すること

採用目安は pooled RMSE 0.10 以上の改善、または 0.05–0.10 でも paired bootstrap と全 fold で一貫し、tail を悪化させない場合とする。0.07 未満の一点差は原則保留する。

## 5. 主力パイプライン

### 5.1 baseline ladder

次の順で実装し、各段階を保存する。

1. 最終既知 TVT を一定に延長（flat baseline）。`surface_anchor` の水平延長は診断用とし、主baselineにはしない
2. 既知区間の surface に robust linear/quadratic fit
3. typewell alignment を使う単一 PF
4. heel-adapted pseudo-typewell + multi-seed PF
5. PF + 小型 residual sequence model
6. PF + residual model + structural projection
7. 上記 + 小さな decorrelated trellis blend

最後の既知 TVT を一定にする flat anchor は全実験の sanity check であり、隠れたリークや座標依存を検出する基準でもある。`TVT+Z` を一定にする予測は、surface表現の恒等式を確認する診断として保持するが、手元データでは大幅に弱いため主baselineには使わない。

### 5.2 heel-adapted pseudo-typewell

予測開始前の GR は同じ well のローカルな地層形状を含む。typewell を完全に置き換えず、heel から推定した profile と blend する。

```text
pseudo_profile = alpha * heel_profile_aligned
               + (1 - alpha) * typewell_profile
```

`alpha` は OOF で固定し、まず 0.5、0.65、0.8 を比較する。報告例では 100 wells の試験で typewell のみ 11.02、0.5 blend 10.31、0.8 blend 9.93、heel のみ 10.19 だった。heel の gain/offset は Huber などでロバストに補正する。

heel 自己相関で datum が約 80% 局在できるという結果は「正しい shape が既知」という条件付きであり、end-to-end 性能の保証ではない。

### 5.3 multi-seed PF

PF の状態は最低限、surface datum と surface rate を持つ。

```text
state_t = [surface_t, rate_t]
surface_t = surface_(t-1) + rate_t * delta_MD + process_noise
rate_t    = rate_(t-1) + small_noise
```

観測 likelihood は GR の pointwise 差だけでなく、局所勾配と短い window の相関を弱く使う。GR を強く信じすぎると self-similar な誤 alignment に吸着するため、continuity を強く、GR evidence を弱くする。

実装順:

- 16 seed で検証ループを作る
- 32、64、128 seed で平均と seed variance の飽和を見る
- likelihood temperature 3/5/8/12 の再重み付けを比較
- PF mean、median、MAP、quantile、seed spread、log-likelihood gap を保存

公開 Notebook の「128×4」は、128 seed を 4 回独立に回す構成ではなく、主に同じ seed path を複数の likelihood temperature で再集約している。計算見積もりでは 512 PF runs と数えない。

### 5.4 小型 residual sequence model

PF 単独の datum/slope 誤りを補う小型 BiGRU または dilated 1D CNN を主候補とする。最初から Transformer や大規模 architecture search は行わない。

入力候補:

- MD、Z、dZ/dMD、trajectory inclination の代理量
- GR、GR の robust z-score、勾配、rolling statistics
- 最終既知点からの距離、予測 horizon
- `surface_anchor`
- PF mean/median/MAP、quantile、spread、likelihood gap
- pseudo-typewell と観測 GR の局所 mismatch

target は `delta_surface` または PF の surface residual とする。損失は Huber を基本に、well ごとの長さが pooled metric に与える影響を再現した版と、well 均等重み版を比較する。

推奨する最初の探索範囲:

- sampling: 2 ft / 4 ft / 8 ft
- hidden width: 64 / 128
- 2–4 層の BiGRU または 5–8 層の dilated CNN
- context: 完全な観測可能系列。ただし mask で known/unknown TVT を区別
- 3–5 seed ensemble

公開 Working Note では多数の Transformer、direct increment、cross-attention が改善せず、best transformer が 8.504 だった一方、CNN + anchor の方が安定した。これは Transformer が不可能という意味ではなく、現在の限られた実験予算では優先度が低いという判断である。

### 5.5 structural projection

モデル出力の surface を、PF とモデルの確信度を重みにして滑らかな軌跡へ射影する。

```text
argmin_S sum_t w_t (S_t - S_model,t)^2
       + lambda1 sum_t (diff(S_t) - rate_prior_t)^2
       + lambda2 sum_t diff2(S_t)^2
```

既知区間との接続は hard constraint または十分大きい重みで固定する。二次 trend は全 well に強制せず、線形・二次・弱平滑の OOF fixed blend を比較する。

### 5.6 decorrelated trellis/path component

弱い whole-well trellis は単体で強くなくても PF と誤差相関が低ければ、小さな blend が効く。報告例では trellis 単体 13.420 でも、誤差相関 0.488 のため blend で 7.762 から 7.699 に改善した。

探索する blend weight は 0.02、0.05、0.10、0.15 程度に限定する。単体 score だけで候補を捨てず、OOF residual correlation と fixed blend の増分で判断する。

## 6. blend、分岐、uncertainty

### 6.1 fixed OOF blend を基本にする

全 OOF 予測を保存し、非負・和 1 の小さな重み探索を行う。ただし最適値へ過適合しないよう、丸い重みを採用する。

```text
final = w_pf * pred_pf
      + w_seq * pred_seq
      + w_path * pred_path
```

初期探索範囲は `w_path <= 0.15`、残りを PF と sequence model に配分する。重みは fold ごとに別にせず、全 OOF で一つに固定する。

### 6.2 selector は原則使わない

候補 oracle が約 6.9 でも learned selector が 8.39–8.71、固定 ensemble が 7.997 だったという報告がある。これは hard well の検出と、どの方向へ直すべきかの判定が別問題だからである。

selector を再検討するのは、以下をすべて満たした場合だけとする。

- 固定 blend より全 fold で改善
- label shuffle で改善が消える
- well 単位 bootstrap で有意
- worst decile と通常 well の双方で悪化しない
- spatial-block でも改善

### 6.3 uncertainty の用途

PF seed spread、GRU seed disagreement、候補間差は error magnitude の検出に使える。しかし signed correction との相関はほぼ 0 という報告がある。

したがって uncertainty は次にだけ使う。

- OOF の tail 分析
- posterior を少し広げる固定 shrinkage
- 近い 2 mode がある場合の限定的な posterior mean
- 第二提出を decorrelate する候補選定

高 uncertainty well に大きな方向付き offset を加える routing は行わない。

約 12% に見られた 15–25 ft の二峰性では midpoint が tail を改善し得るが、global midpoint にはしない。near-tie、十分な mode separation、OOF tail 改善という 3 条件を満たす場合だけ実験する。

## 7. 空間・近隣 well の扱い

主催者コメントから、近隣 well の formation dip が似ること自体は妥当である。一方で、通常 GroupKFold で 7.699→7.635 と改善した spatial model が公開 LB で 6.765→7.002 に悪化し、leave-spatial-block では改善が -0.013 まで消えた例がある。

したがって以下は主力から隔離する。

- nearest-well profile の直接コピー
- 座標から datum/slope を回帰
- azimuth group ごとの補正
- field-wide Gaussian Process
- typewell duplicate の直接転写

特に duplicate typewell transfer は RMSE 約 117、anchor baseline 約 18.1 という破滅的失敗の報告がある。同じ形に見えても datum が同じとは限らない。

近隣距離 150 ft 未満で RMSE 6.70、600 ft 超で 10.15 という Discussion の集計は有望な層別結果だが、近隣転写の因果的効果を示さない。Stage 5 の独立実験として spatial-block gate を通った場合だけ採用する。

## 8. same-well contact override の隔離

公開 Notebook では sample test の 3 wells が train に同じ ID で存在し、contact override が train の真の TVT から offset を計算する。これにより 3 wells で RMSE 約 0.005–0.010 となる一方、通常モデル部分は約 6.47 だったという検証がある。

これは物理モデルの精度ではなく、答えの代入に近い経路である。次のルールを設ける。

1. honest CV と core model の OOF では完全に無効化
2. score や ablation に override 後の値を混ぜない
3. 同一 ID、重複度、参照した target 列を audit log に残す
4. 主提出には原則使わない
5. 使う場合は競技固有の別提出として明記し、ルール適合性と hidden test の一致条件を別途確認

「visible prefix が一致したので悪化しない」とは言えない。hidden 側の同一 ID が同じ版・同じ datum である保証はなく、public sample の挙動だけでは一般化を評価できない。

## 9. 実験ロードマップ

### Stage 0 — honest evaluation harness

- target mask と予測区間を公式形式に合わせる
- whole-well fold と人工 cut の lineage を固定
- surface target と TVT reconstruction の単体 test
- train/test ID overlap と duplicate fingerprint のレポート
- pooled、per-well、tail、bias、slope 指標を一括出力

完了条件: flat anchor と既存公開 PF を同一 harness で評価できる。

### Stage 1 — surface baselines と PF

- flat / linear / quadratic surface baseline
- typewell PF
- heel pseudo-profile の 0.5/0.65/0.8 blend
- PF 16 seed、likelihood 温度の小探索

判断点: heel blend が全置換より良く、複数 fold と tail で一貫するか。

実装済みの事前trend検証では、linear/quadratic TVT・surfaceはいずれも単体でanchorを大幅に下回った。ただし last-known TVT 95%、linear surface 4%、quadratic surface 1% の固定blendは全5 foldsで改善し、pooled RMSE 15.909853→15.700508、nested weight selectionでも15.742048となった。この成分は単体モデルではなく、PF前の弱いbias補正としてのみ保持する。

### Stage 2 — PF の安定化

- 32/64/128 seed の収束曲線
- robust likelihood、gradient evidence、continuity prior
- mean/median/MAP と small fixed blend
- quadratic/robust surface projection

判断点: 追加計算が 0.10 以上の再現可能な改善を生むか。

初回実装では、256 particles × 16 seeds、likelihood温度5のtypewell PFを、正解TVTを予測入力へ渡さない形で再実装した。全773 wellsのOOFでseed 42はRMSE 13.178118となり、Stage 1の15.700508から全5 foldsで改善した。一方seed 43は13.930864で、16 seedsでは不安定だった。そこで独立な2×16-seed batchesを平均するとPF単体12.899535となった。fold外で重みを選ぶnested検証ではPF重みが0.75または0.80に揃い、nested RMSE 12.597170だった。このためpromoted構成は2-batch PF 75% + Stage 1 guarded trend blend 25%とし、同一OOFでRMSE 12.565438を得た。2-batch PF単体に対するwell単位paired bootstrapの95%区間は -0.292〜-0.010である。次の探索はparticlesやseed数の単純増加より、heel pseudo-profileとlikelihood evidenceの改善を優先する。

### Stage 3 — residual sequence model

- 1D CNN と BiGRU を各 1–2 構成
- target は delta_surface と PF residual
- full sequence + mask、anchor 明示
- 3 seed ensemble
- PF との OOF residual correlation を確認

判断点: 単体 score でなく PF との final fixed blend が改善するか。

### Stage 4 — path diversity と tail

- 弱い trellis/path posterior
- 2–15% の fixed blend
- bimodal near-tie に限った midpoint 実験
- uncertainty bucket ごとの calibration と tail 分析

判断点: worst decile を改善し、通常 well を悪化させないか。

### Stage 5 — 空間実験（時間がある場合のみ）

- 距離、方位、近隣 slope の最小特徴セット
- GroupKFold と leave-spatial-block-out の二重評価
- shuffle / boundary audit

gate を一つでも落としたら最終候補に入れない。

## 10. 優先度を下げる実験

- 大型 Transformer、cross-attention、長期 architecture sweep
- 全候補を動的に切り替える selector
- unrestricted DTW/NCC の直接 alignment
- 同じ path を再重み付けするだけの過剰な PF 計算
- 独立検証なしに 3 個以上の post-processing を積むこと
- 単純な shrink-to-last-known
- 座標だけから datum/slope を当てるモデル
- public sample の contact override を core model として最適化
- 非公開の「CV < 5」主張から architecture を推測すること

## 11. 実験台帳

各 run で以下を保存する。

```text
run_id
code_hash
data_fingerprint
fold_definition_hash
seed
feature_set
model_config
postprocess_stack
pooled_rmse
fold_rmse
well_median_rmse
well_p90_rmse
worst_5pct_sse_share
mean_bias_rmse
slope_error
paired_bootstrap_ci
runtime
artifact_paths
```

PF の cluster rerun で同一 pipeline が公開 7.096 / 7.135 / 7.091 と揺れた報告があるため、コード hash とデータ fingerprint が同じ run でも複数 seed・再実行を残す。

## 12. 最終提出方針

提出枠は目的を分ける。

### 提出 A — conservative public-proven

公開 Notebook で再現可能な PF/beam/learned-path 系の保守設定を基にする。確認できた snapshot では、`rogii-det-mha140sep4` が public 6.979、`rogii-public-7-061-exact-reproduction` はタイトルとは異なりページ表示 7.099 だった。数値は参考であり、sample contact override を除いた honest OOF を別途確認する。

### 提出 B — private-expectation

この文書の honest GroupKFold + spatial audit を通った surface PF、residual sequence model、small trellis の decorrelated fixed blend を使う。public 値への追従より、fold・seed・tail の一貫性を優先する。

2 提出をほぼ同じモデルにせず、OOF residual correlation を見て意味のある差を残す。締切直前に public LB を見て selector や補正を追加しない。

## 13. 現実的な目標

公開資料から再現性を確認できる範囲はおおむね public 6.98–7.10 であり、Working Note の best は 6.675 や 6.794 と報告されている。一方、5 点台の per-well-only model は設計非公開のコメントであり、現時点では再現目標に使えない。

したがって目標を次の順に置く。

1. honest OOF で既存 PF family を再現
2. seed/fold をまたいで 0.2–0.4 改善する residual blend を作る
3. worst 5–10% wells の SSE を減らす
4. その結果として 6 点台前半以下を狙う

金メダル水準は挑戦目標として残すが、根拠のない累積改善幅を足し合わせて「5.5 予測」とはしない。

## 14. 参照資料

ローカル保存済み Discussion:

- [Working Note Winners](../discussion/raw/727171.json)
- [Discussion / writeup link collection](../discussion/raw/716699.json)
- [Fork the ruler, not the model](../discussion/raw/712037.json)
- [Self-similarity and bimodal wells](../discussion/raw/711878.json)
- [Host hints on nearby wells and lateral GR](../discussion/raw/698825.json)
- [Parallel formations / truncated ANCC](../discussion/raw/708167.json)
- [Distance-stratified validation discussion](../discussion/raw/726465.json)
- [Architecture score discussion](../discussion/raw/717573.json)
- [Outlier well exclusion](../discussion/raw/707695.json)
- [Public/private CV divergence](../discussion/raw/704273.json)
- [Shared failure segments](../discussion/raw/726834.json)
- [Guarded geosteering approach](../discussion/raw/717445.json)

外部資料:

- [When Better CV Scores Worse — A Control-First Geosteering Study](https://www.kaggle.com/writeups/radiantallomancer/when-better-cv-scores-worse-a-control-first-geost)
- [The Wiggle Is Free, the Trend Is the Wall](https://www.kaggle.com/writeups/malyshevdanil/the-wiggle-is-free-the-trend-is-the-wall)
- [ROGII DET MHA140SEP4](https://www.kaggle.com/code/canqiang/rogii-det-mha140sep4)
- [ROGII Public 7.061 Exact Reproduction](https://www.kaggle.com/code/kaitofukami/rogii-public-7-061-exact-reproduction)

公開スコア、Notebook の表示内容、Discussion の状態は 2026-07-19 時点の snapshot である。
