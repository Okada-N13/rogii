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
