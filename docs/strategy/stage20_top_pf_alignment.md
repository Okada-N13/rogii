# Stage 20: top-PF-aligned validation

## 背景

Stage 19A/BはStage 17 strong-base代理上で全gateを通過したが、実際の6.589 top-PF baseへ
weight 0.50、cap 16 ftの補正を加えたStage 19CはPublic LB `6.958`だった。基準`6.589`から
`+0.369`の明確な悪化であり、Stage 19Cを棄却する。

この結果は、モデルの3係数表現より先にbase alignmentが不十分だったことを示す。
Stage 17 baseはpublic OOFと軽量selectorのhybridであり、470には次の追加構造がある。

- A130: visible-prefix GR likelihood width `×1.30`
- SP45 ridge/selector `30/70`
- U-space robust polynomial projection
- SP45/learned branch `60/40`
- visible-prefix adaptive candidate
- tiny model-package correction
- conservative PF seed-branch hedge

## 完全OOFが作れない理由

470のlearned branchは全train wellsで学習済みの公開model packageであり、各wellを除外したfold modelは
公開されていない。このmodelをtrain pseudo-testへ直接適用すると教師を見たmodelによるleakになる。
したがって「470完全OOF」を名乗らない。

Stage 20Aではlearned branchをStage 17Aのwell-isolated public OOFで代用する。visible-prefix adaptive
candidate、tiny model-package、seed-branch hedgeは省略する。これは470そのものではなく、
Stage 17より470に近いtarget-safe proxyである。

## Stage 20A固定screen

対象はStage 17Aでpublic OOFを安全にreplayできるprimary cutsだけとする。5 standard folds ×
5 prefix fractions × 8 cutsをcut IDのSHA-256で固定し、計200 cutsを使う。

各cutで以下を構成する。

```text
public = well-isolated public OOF
selector = A130 likelihood PF
sp45_raw = 0.30 * public + 0.70 * selector
sp45_projected = U-space robust degree-3 projection (75%)
top_pf_proxy = 0.60 * sp45_projected + 0.40 * public
```

このbaseからStage 19と同じ3係数targetを作る。66特徴はvisible prefix、全MD/X/Y/Z/GR、
typewell、base predictionだけから作り、hidden suffix TVTは教師と評価にしか使用しない。

Stage 19C失敗後の固定profileはweight `0.10`、cap `8 ft`、ramp `96 rows`。weight
`0.05/0.10/0.20/0.35/0.50`は診断表として出すが、同一screen上の最良weightを直接提出しない。

Gate:

- hidden-target invariance
- public OOF provenance
- standard RMSE `-0.05`以上
- well bootstrap 95%上限 `< 0`
- standard foldの80%以上で改善
- spatial/typewell/branch-groupの全体RMSE改善
- well P90非悪化

全gate通過時だけStage 20Bへ進み、全eligible cutsと別のshort-prefix proxyで確認する。
Stage 20AからKaggle submissionは作らない。

## 実行

Colab CPUで次を単独実行する。

`notebooks/520_run_stage20a_top_pf_alignment.ipynb`

必要artifact:

- `stage16b_testlike_validation_full_v003`
- `stage17_public_replay_full_v002`
- competition train data

途中で25 cutsごとに進捗を表示する。最後の辞書とweight診断表を共有する。

## Stage 20A実測

194 cuts、158 wells、1,114,472 suffix rowsで実行した。200未満なのは一部stratumに8 cuts
存在しなかったためで、選択規則は変更していない。

- public OOF: `12.1803`
- A130 selector: `10.3820`
- top-PF proxy: `9.9810`
- 固定weight 0.10: `9.9810 → 9.8611`（`-0.1199`）
- standard/spatial/typewell/branch-groupはすべて全体改善
- standard 5/5 folds、5/5 fractions改善
- bootstrap 95%: `[-0.1044, +0.00625]`
- well P90: `+0.1006`悪化

bootstrap上限とwell P90で不合格のため、事前固定weight 0.10はStage 20Bへ昇格しない。
weight 0.35/0.50はpooled gainが大きくてもfold consistencyが3/5、2/5へ崩れ、
Stage 19C weight 0.50のLB悪化と整合する。

診断weight 0.05は5/5 folds、5/5 fractions、cut P90/max、worst-tailを改善し、
well P90悪化が`+0.0084`だけだった。ただし結果を見て選んだ値なので、同じsampleで昇格させない。

## Stage 20B: disjoint-well confirmation

Stage 20Aの`cut_features.parquet`に含まれる158 wellsを候補poolから先に完全除外する。
残った別wellから同じfold × fraction規則で固定sampleを作る。cutの重複だけでなくwell重複をゼロにする。

Stage 20Bのprimary profileはweight `0.05`、cap `8 ft`、ramp `96 rows`に事前固定した。
診断weightは昇格判断へ使わない。追加gateとして`discovery_well_overlap_zero`を必須にする。

Colab CPU:

`notebooks/530_run_stage20b_disjoint_confirmation.ipynb`

Stage 20Bが全gateを通過した場合も直接提出しない。Stage 20Cで全remaining eligible cutsと、
public OOFを安全にreplayできないshort-prefixに対する独立proxyを確認する。不合格なら
6.589 base向け3係数trajectory residualを終了する。

## Stage 20B実測と終了判断

Stage 20Aの158 wellsを完全除外し、別139 wells・163 cuts・842,785 rowsで確認した。
discovery well overlapはゼロだった。

- base: `9.49547`
- 固定weight 0.05: `9.46591`（`-0.02956`）
- standard 5/5 folds、5/5 fractions改善
- bootstrap 95%: `[-0.05161, +0.00350]`
- well P90: `-0.1719`
- spatial: `+0.00921`悪化
- typewell: `+0.02826`悪化
- branch-group: `-0.06319`改善

minimum gain `0.03`、bootstrap、spatial、typewellで不合格。診断weight 0.10はstandard gainを
増やすが、結果を見た後の選択であり、distribution robustnessも解決しない。

Stage 19CのLB悪化、Stage 20A/Bの独立確認を合わせ、6.589 base向け3係数trajectory residualを
終了する。Stage 20Cは実施しない。次はvisible-prefix内の実測backtestから候補pathを選ぶ
Stage 21 candidate routingへ移る。
