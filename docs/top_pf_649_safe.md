# Top-PF 6.49 target-safe reproduction

`notebooks/470_kaggle_top_pf_a130_branch_safe.ipynb`は、公開Notebook`top-pf-config-branch-conservative(1).ipynb`の申告スコア6.49を、安全なV599 controlから再現する候補である。6.49はユーザーが確認した公開情報であり、この派生Notebook自身の実測値ではない。

## Hidden-testで有効な差分

実測6.685の`230_kaggle_v599_a130_frontier_safe.ipynb`に対して、予測差を次の2点へ限定する。

- learned PFのGR likelihood scaleを`1.3`倍
- branch hedgeをstrength `0.60`、cap `2.0`、既存routeをskipしない設定へ変更

後者だけを変更した過去の安全版は実測6.693だった。したがって、申告6.49の主要仮説はPF GR likelihood scale `×1.3`である。

## 除外した処理

公開3 fake wellsはtrainからの例示コピーである。元Notebookは同じwell IDがtrainにある場合、trainの完全な`TVT`からcontact trajectoryを作りtestを上書きする。実hidden wellsでは同じ条件を期待できず、手法評価も歪めるため以下を含めない。

- `tvt_from_contacts`
- `hw_tr['TVT']`を使うsame-well transfer
- guarded/contact override
- overlap dry-run probe
- contact candidateのvisible-prefix再適用
- LB probeまたはleaderboard-derived bias
- submissionを書き換えない重いCV診断

## 提出順序

現在進行中のStage 18 v003結果を先に確定する。その後、このNotebookをStage 18なしの単独候補として1回提出し、6.685 controlと比較する。単独再現を確認する前にStage 18と混ぜない。

Kaggleでは既存V599と同じInputs、Internet OFF、P100を使用する。最終`V599_FRONTIER_SAFE_AUDIT`が14,151 rows、ID順一致、finite TVT、profile `top_pf_a130_branch_conservative_safe`を示すことを確認する。
