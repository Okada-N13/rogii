# Stage 21: visible-prefix candidate routing

## 目的

Stage 19/20の3係数残差は平均的には小さく改善したが、spatial/typewell分布で符号が安定せず、
実LBを`6.589 → 6.958`へ悪化させた。Stage 21ではhidden residualを特徴から直接推定しない。
test well自身の既知`TVT_input` prefix内に擬似cutを作り、複数候補を実測backtestして選ぶ。

## Stage 21A discovery split

Stage 20Aの158 wellsとStage 20Bの139 wellsをすべて除外する。残った別wellのうち、
well-isolated public OOFの元cutからouter cutまで96 rows以上あるprimary cutsを使う。
standard fold × requested fractionごとに最大4 cutsをSHA-256固定抽出する。

各outer cutについて、元public OOF cutとouter cutの間に2つのinternal cutを固定する。
internal cutからouter cut直前までのTVTは、実test時点ではvisible prefixなのでbacktest教師として使用可能。
outer cut以降のTVTは候補生成・score・選択に使わない。

## 候補library

- well-isolated public OOF
- selector A100 / A130 / A160
- top-PF proxy A100 / A130 / A160
- prefix Uのrobust polynomial degree 1 / 2 / 3

計10候補。PF screen解像度は64 particles、4 seeds、最大384 steps。各候補の2 internal
holdout RMSEの中央値をscoreとし、最小候補を選ぶ。

primary outputは選択候補そのものではない。top-PF A130 proxyをbaseとし、
選択候補との差をweight `0.25`、cap `12 ft`、ramp `96 rows`でguarded blendする。
これらはStage 21A結果を見る前に固定する。

## Gate

- hidden-target invariance
- public OOF target-safe
- Stage 20A/B discovery well overlapゼロ
- internal score rankとouter error rankの相関 `>= 0.10`
- guarded router RMSE gain `>= 0.05`
- well bootstrap 95%上限 `< 0`
- standard / fraction / spatial / typewell / branch-group consistency
- well P90非悪化

oracle candidate RMSE、raw router RMSE、top1 oracle一致率は診断であり昇格条件を後付け変更しない。

全gate通過時だけ、別wellかつ高PF解像度のStage 21Bへ進む。Stage 21Aからlearned routerや
Kaggle submissionを作らない。

## Stage 21A実測結果

2026-07-24に77 cuts・63 wellsで完了したが棄却した。base RMSE `9.1901`に対して
guarded routerは`9.8522`（`+0.6621`）、bootstrap 95%は`[+0.1895,+1.7333]`、
standard fold改善は1/5だった。internal/outer rank correlationは`0.2754`だが、
top-1 oracle一致率は`5.19%`に留まった。raw router RMSE `450.64`は多項式候補の
catastrophic extrapolationによる。oracle RMSE `5.4550`という候補多様性はあるが、
2点のinternal scoreから直接候補を選ぶ規則には一般化能力がなかった。

## Stage 21B disjoint confidence gate

Stage 21Aの63 wellsは候補固有のinner→outer楽観バイアス校正にのみ使用する。Stage 20A/Bと
Stage 21Aの全wellを除いた別sampleを評価するため、Stage 21A結果上での再最適化ではない。

- degree 1/2/3の多項式候補をすべて除外する。
- public OOF、selector A100/A130/A160、top-PF A100/A130/A160の7候補だけを使う。
- Stage 21Aで候補別の`outer_rmse - inner_rmse`中央値とrobust MADを推定する。
- 新sampleではrisk-adjusted scoreを使い、A130より補正後`0.50`以上良いことを要求する。
- 2 internal cutsの両方でA130より良い場合だけ代替候補を受理する。
- primary correctionはweight `0.10`、cap `8 ft`、ramp `96 rows`に固定する。
- diagnostic weight `0.05/0.10/0.15/0.20`は報告するがprimary gateを後付け変更しない。

Stage 21Bもhidden-target invariance、bootstrap、standard/fraction/spatial/typewell/
branch-group consistency、well P90をすべて通過した場合だけStage 21Cへ進む。不通過なら
prefix candidate routing自体を終了する。

## 制約

470のfold-safe learned branch modelは公開されていないため、well-isolated public OOFで代用する。
また470既存のvisible-prefix overlayはproxy baseへ含まれない。このためStage 21Aは470完全OOFではない。
この制約はStage 20と同様に結果解釈へ残す。

## 実行

Colab CPU:

`notebooks/540_run_stage21a_prefix_router.ipynb`

Stage 21B:

`notebooks/550_run_stage21b_prefix_confidence.ipynb`

必要artifact:

- `stage16b_testlike_validation_full_v003`
- `stage17_public_replay_full_v002`
- `stage7_public_residual_gate_full_v001`
- `stage20a_top_pf_alignment_full_v001`
- `stage20b_disjoint_confirmation_full_v001`

10 cutsごとに進捗を表示する。最後のsummary辞書とcandidate集計表を共有する。
